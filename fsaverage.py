# %%
import numpy as np
import igl
import utils
import matplotlib.pyplot as plt
import os
from nilearn import datasets, surface

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

    fsaverage3 = datasets.fetch_surf_fsaverage("fsaverage3")
    fsaverage4 = datasets.fetch_surf_fsaverage("fsaverage4")

    bending_weight = 1e-1
    kmin = 5
    kmax = 100
    precise = False  # convert the final vertex map to a vertex-to-point map
    exists_gt = False  # groundtruth exists

    vA, fA = surface.load_surf_mesh(fsaverage4["pial_left"])
    # Offset the left hemisphere by 10 units in x direction
    vA[:, 0] -= 100

    # Rotate the left hemisphere by 90 degrees around the x-axis

    vB, fB = surface.load_surf_mesh(fsaverage3["pial_left"])

    featuresA = np.ones((vA.shape[0], 5))
    featuresB = np.ones((vB.shape[0], 5))

    shapeA = Shape(vA, fA, name=nameA)
    shapeB = Shape(vB, fB, name=nameB)

    # compute correspondences using the elastic eigenmodes (elasticBasis)
    Solv_elastic = CorrespondenceSolver(
        shapeA,
        shapeB,
        featuresA=featuresA.T,
        featuresB=featuresB.T,
        kmin=kmin,
        kmax=kmax,
        bending_weight=bending_weight,
        elasticBasis=True,
    )
    # compute final correspondences by an iterative procedure
    P, C = Solv_elastic.computeCorrespondence()

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
        shapeA.normals = featuresA.T
        shapeB.normals = featuresB.T
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

    ps.show()
