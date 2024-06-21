# %%
import numpy as np
import igl
import utils
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from nilearn import datasets, surface, image, plotting
from scipy.sparse import csr_matrix

from correspondence import Shape
from correspondence import CorrespondenceSolver

try:
    ps = __import__("polyscope")
    print("Visualization with polyscope.")
    visualization = True
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
    fsaverage = datasets.fetch_surf_fsaverage("fsaverage6")

    def load_images_and_project_to_surface(image_paths, fsaverage):
        """Util function for loading and projecting volumetric images."""
        images = [image.load_img(img) for img in image_paths]
        surface_images = [
            np.nan_to_num(surface.vol_to_surf(img, fsaverage.pial_left))
            for img in images
        ]

        return np.stack(surface_images)

    featuresA = load_images_and_project_to_surface(
        source_imgs_paths, fsaverage
    ).T
    featuresB = load_images_and_project_to_surface(
        target_imgs_paths, fsaverage
    ).T

    vA, fA = surface.load_surf_mesh(fsaverage["pial_left"])
    vB, fB = np.copy(vA), np.copy(fA)
    # Offset the left hemisphere by 10 units in x direction
    vA[:, 0] -= 100

    # Rotate the left hemisphere by 90 degrees around the x-axis

    # vB, fB = surface.load_surf_mesh(fsaverage3["pial_left"])

    shapeA = Shape(vA, fA, name=nameA)
    shapeB = Shape(vB, fB, name=nameB)

    bending_weight = 1e-1
    kmin = 1
    kmax = 30
    precise = False  # convert the final vertex map to a vertex-to-point map
    exists_gt = False  # groundtruth exists
    # compute correspondences using the elastic eigenmodes (elasticBasis)
    Solv_elastic = CorrespondenceSolver(
        shapeA,
        shapeB,
        kmin=kmin,
        kmax=kmax,
        bending_weight=bending_weight,
        elasticBasis=False,
    )

    # %%
    # compute final correspondences by an iterative procedure
    P, C = Solv_elastic.computeCorrespondence()
    # Refine the correspondence using the features
    P, C = Solv_elastic.refineCorrespondence(
        C,
        featuresA,
        featuresB,
        alpha=100,
        nits=10000,
    )

    # convert mapping matrix to indices vA->vB[corr_ours]
    # Assuming P is a sparse matrix in CSR format
    P = csr_matrix(P)

    # Get the non-zero indices in the transposed matrix
    rows, cols = P.T.nonzero()
    corr_ours = cols

    # %%

    # %%
    # Visualize the correspondence with nilearn
    projected_features = featuresB[corr_ours]

    fig, axes = plt.subplots(
        2, 2, subplot_kw={"projection": "3d"}, figsize=(12, 10)
    )
    fig.suptitle("Functional mapping", fontsize=16)

    # Plot the source features
    plotting.plot_surf_stat_map(
        fsaverage.pial_left,
        featuresB[:, -1],
        hemi="left",
        bg_map=fsaverage.sulc_left,
        view="lateral",
        cmap="coolwarm",
        axes=axes[0, 0],
        colorbar=True,
        title="Source features",
    )

    # Plot the projected features
    plotting.plot_surf_stat_map(
        fsaverage.pial_left,
        projected_features[:, -1],
        hemi="left",
        bg_map=fsaverage.sulc_left,
        view="lateral",
        cmap="coolwarm",
        axes=axes[0, 1],
        colorbar=True,
        title="Projected features",
    )

    # Plot the target features
    plotting.plot_surf_stat_map(
        fsaverage.pial_left,
        featuresA[:, -1],
        hemi="left",
        bg_map=fsaverage.sulc_left,
        view="lateral",
        cmap="coolwarm",
        axes=axes[1, 0],
        colorbar=True,
        title="Target features",
    )

    # Plot the difference between the source and projected features
    plotting.plot_surf_stat_map(
        fsaverage.pial_left,
        featuresB[:, -1] - projected_features[:, -1],
        hemi="left",
        bg_map=fsaverage.sulc_left,
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

    # %%
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
        normal_diff = shapeB.normals[corr_ours] - shapeA.normals
        target_mesh.add_color_quantity("normals", shapeB.normals, enabled=True)
        source_mesh.add_color_quantity(
            "elastic Basis pullback normals",
            normal_diff,
            enabled=True,
        )
        target_mesh.set_position(np.array([0, 0, 1]))

        ps.show()
