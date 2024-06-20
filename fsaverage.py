# %%
import numpy as np
import igl
import utils
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from nilearn import datasets, surface, image, plotting

from correspondence import Shape
from correspondence import CorrespondenceSolver

try:
    ps = __import__("polyscope")
    print("Visualization with polyscope.")
    visualization = False
except ImportError:
    print("Module polyscope not found. No visualization of result.")
    visualization = False

if __name__ == "__main__":
    readPath = "data/"
    writePath = "results/"
    if not os.path.exists(writePath):
        os.mkdir(writePath)

    nameA = "fsaverage_lh"
    nameB = "fsaverage_rh"

    n_subjects = 2

    contrasts = [
        # "sentence reading vs checkerboard",
        "sentence listening",
        # "calculation vs sentences",
        # "left vs right button press",
        # "checkerboard",
    ]

    brain_data = datasets.fetch_localizer_contrasts(
        contrasts,
        n_subjects=n_subjects,
        get_anats=True,
    )

    source_imgs_paths = brain_data["cmaps"][0 : len(contrasts)]
    target_imgs_paths = brain_data["cmaps"][
        len(contrasts) : 2 * len(contrasts)
    ]
    # fsaverage3 = datasets.fetch_surf_fsaverage("fsaverage3")
    # fsaverage4 = datasets.fetch_surf_fsaverage("fsaverage4")
    fsaverage6 = datasets.fetch_surf_fsaverage("fsaverage6")

    def load_images_and_project_to_surface(image_paths, fsaverage):
        """Util function for loading and projecting volumetric images."""
        images = [image.load_img(img) for img in image_paths]
        surface_images = [
            np.nan_to_num(surface.vol_to_surf(img, fsaverage.pial_left))
            for img in images
        ]

        return np.stack(surface_images)

    featuresA = load_images_and_project_to_surface(
        source_imgs_paths, fsaverage6
    ).T
    featuresB = load_images_and_project_to_surface(
        target_imgs_paths, fsaverage6
    ).T

    print(featuresA.shape)
    print(featuresB.shape)

    # fsaverage4 = datasets.fetch_surf_fsaverage("fsaverage4")

    bending_weight = 1e-1
    kmin = 1
    kmax = 30
    precise = False  # convert the final vertex map to a vertex-to-point map
    exists_gt = False  # groundtruth exists

    vA, fA = surface.load_surf_mesh(fsaverage6["pial_left"])
    vB, fB = np.copy(vA), np.copy(fA)
    # Offset the left hemisphere by 10 units in x direction
    vA[:, 0] -= 100

    # Rotate the left hemisphere by 90 degrees around the x-axis

    # vB, fB = surface.load_surf_mesh(fsaverage3["pial_left"])

    shapeA = Shape(vA, fA, name=nameA)
    shapeB = Shape(vB, fB, name=nameB)

    # %%

    # compute correspondences using the elastic eigenmodes (elasticBasis)
    Solv_elastic = CorrespondenceSolver(
        shapeA,
        shapeB,
        kmin=kmin,
        kmax=kmax,
        bending_weight=bending_weight,
        elasticBasis=True,
    )

    print(shapeB.normals.shape)

    # compute final correspondences by an iterative procedure
    P, C = Solv_elastic.computeCorrespondence()

    # %%
    # Refine the correspondence using the features
    P, C = Solv_elastic.refineCorrespondence(
        C,
        featuresA,
        featuresB,
        alpha=0.0,
        nits=10000,
    )

    # convert mapping matrix to indices vA->vB[corr_ours]
    corr_ours = P.toarray()
    corr_ours = np.nonzero(corr_ours.T)[1]
    np.savetxt(
        writePath
        + shapeA.name
        + "_"
        + shapeB.name
        + "_ElasticBasisresult.txt",
        corr_ours,
        fmt="%d",
    )

    if precise:
        P_prec = Solv_elastic.preciseMap(C)
        np.save(
            writePath
            + shapeA.name
            + "_"
            + shapeB.name
            + "ElasticPrecisemap.npy",
            P_prec,
            allow_pickle=True,
        )

    # use the eigenfunctions of LB operator as a comparison (this method corresponds to ZoomOut)
    Solv_LB = CorrespondenceSolver(
        shapeA, shapeB, kmin=kmin, kmax=kmax, LB=True
    )

    P_LB, C_LB = Solv_LB.computeCorrespondence()

    # Refine the correspondence using the features
    P_LB, C_LB = Solv_LB.refineCorrespondence(
        C,
        featuresA,
        featuresB,
        alpha=0.0,
        nits=10000,
    )

    # #convert mapping matrix to indices vA->vB[corr_LB]
    corr_LB = P_LB.toarray()
    corr_LB = np.nonzero(corr_LB.T)[1]

    if precise:
        P_prec_LB = Solv_LB.preciseMap(C_LB)
        np.save(
            writePath + shapeA.name + "_" + shapeB.name + "_LBPrecisemap.npy",
            P_prec_LB,
            allow_pickle=True,
        )

    np.savetxt(
        writePath + shapeA.name + "_" + shapeB.name + "_LBBasisresult.txt",
        corr_LB,
        fmt="%d",
    )
    print("saved computed correspondence in result folder")
    ########visualize results#######
    if visualization:
        ps.init()

        source_mesh = ps.register_surface_mesh(
            "source shape", shapeA.v, shapeA.f, smooth_shade=True
        )
        target_mesh = ps.register_surface_mesh(
            "target shape", shapeB.v, shapeB.f, smooth_shade=True
        )

        # normal transfer
        target_mesh.add_color_quantity("normals", shapeB.normals, enabled=True)
        source_mesh.add_color_quantity(
            "elastic Basis pullback normals",
            shapeB.normals[corr_ours],
            enabled=True,
        )
        source_mesh.add_color_quantity(
            "LB Basis pullback normals", shapeB.normals[corr_LB], enabled=False
        )
        target_mesh.set_position(np.array([0, 0, 1]))

        ps.show()
    if precise:
        source_mesh.add_color_quantity(
            "elastic precise map pullback normals",
            P_prec.dot(shapeB.normals),
            enabled=True,
        )
        source_mesh.add_color_quantity(
            "LB precise map pullback normals",
            P_prec_LB.dot(shapeB.normals),
            enabled=False,
        )

    # %%
    # Visualize the correspondence with nilearn
    projected_features = featuresB[corr_LB]

    fig, axes = plt.subplots(
        2, 2, subplot_kw={"projection": "3d"}, figsize=(12, 10)
    )
    fig.suptitle("Functional mapping", fontsize=16)

    # Plot the source features
    plotting.plot_surf_stat_map(
        fsaverage6.pial_left,
        featuresB[:, -1],
        hemi="left",
        bg_map=fsaverage6.sulc_left,
        view="lateral",
        cmap="coolwarm",
        axes=axes[0, 0],
        colorbar=True,
        title="Source features",
    )

    # Plot the projected features
    plotting.plot_surf_stat_map(
        fsaverage6.pial_left,
        projected_features[:, -1],
        hemi="left",
        bg_map=fsaverage6.sulc_left,
        view="lateral",
        cmap="coolwarm",
        axes=axes[0, 1],
        colorbar=True,
        title="Projected features",
    )

    # Plot the target features
    plotting.plot_surf_stat_map(
        fsaverage6.pial_left,
        featuresA[:, -1],
        hemi="left",
        bg_map=fsaverage6.sulc_left,
        view="lateral",
        cmap="coolwarm",
        axes=axes[1, 0],
        colorbar=True,
        title="Target features",
    )

    # Plot the difference between the source and projected features
    plotting.plot_surf_stat_map(
        fsaverage6.pial_left,
        featuresB[:, -1] - projected_features[:, -1],
        hemi="left",
        bg_map=fsaverage6.sulc_left,
        view="lateral",
        cmap="coolwarm",
        axes=axes[1, 1],
        colorbar=True,
        title="Difference between source and projected features",
        # Threshold at 10% of the maximum value
        threshold=3,
    )

    # plt.tight_layout(
    #     rect=[0, 0.03, 1, 0.95]
    # )  # Adjust layout to make room for the title
    plt.show()
