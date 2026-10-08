# Third-Party Notices

This release contains code derived from or distributed with the following
projects. Each component remains subject to its original license and
attribution requirements.

## FreeGave

- Project: <https://github.com/vLAR-group/FreeGave>
- Paper: *FreeGave: 3D Physics Learning from Dynamic Videos by Gaussian Velocity*
- License: see `LICENSE` at the repository root.

## 3D Gaussian Splatting rasterizer

- Source family: <https://github.com/graphdeco-inria/gaussian-splatting>
- Local license: `submodules/depth-diff-gaussian-rasterization/LICENSE.md`
- The local license restricts use to non-commercial research and evaluation.

## GLM

- Vendored under the rasterizer's `third_party/glm` directory.
- Local license: `submodules/depth-diff-gaussian-rasterization/third_party/glm/copying.txt`

## simple-knn

- Original source: <https://gitlab.inria.fr/bkerbl/simple-knn>
- Source headers identify Inria GRAPHDECO and refer to the Gaussian Splatting
  non-commercial research/evaluation license. Preserve those headers and review
  the upstream terms before redistribution.

## Other acknowledgements

FreeGave builds on Deformable 3D Gaussians and NVFi. See the main README and
the original FreeGave publication for full citations.

