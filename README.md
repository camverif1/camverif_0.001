# camverif_0.001


Requirements
------------------
 - **`z3`**: install z3 python.
 - **`pyparma ppl`**: (https://pypi.org/project/pyparma/).

Setup Scene
-----------------
The scene contains a set of objects, and each object is defined using triangle meshes. Each triangle has three vertices, and each vertex has an associated RGB color.


## Scene Definition

Below, we describe how a 3D scene with objects is defined in the `scene.py` file. Each object in the scene is modeled using triangle meshes.

### Variables in `scene.py`

- **`numOfTriangles`**: Specifies the number of triangles in the scene.
- **`numOfVertices`**: Indicates the total number of vertices in the scene.
- **`numOfEdges`**: Specifies the total number of edges in the scene.

### Data Structures

- **`vertices`**: A list containing the 3D coordinates of each vertex in the global coordinate system.
- **`vertColours`**: A list containing the RGB color for each vertex, with three entries per vertex (one for red, one for green, and one for blue).
- **`nvertices`**: A list of vertex indices for each triangle, with three entries per triangle (one index for each vertex).
- **`tedges`**: A list containing the indices of vertices that form the edges of each triangle, where each edge is defined by two vertex indices (one for each endpoint).

### Example

Suppose we have a scene with a single rectangular object. Number the vertices of the rectangle as 0, 1, 2, and 3.

#### Vertex Coordinates

The `vertices` list contains the 3D coordinates of each corner of the rectangle:

```python
vertices = [1,1,1, 2,1,1, 2,2,1, 1,2,1]
```

#### Vertex Colors

Assume each vertex is green. The `vertColours` list is:

```python
vertColours = [0,1,0, 0,1,0, 0,1,0, 0,1,0]
```

#### Triangles

To form triangles from the rectangle, we divide it into two triangles:

- **Triangle `t1`** is formed by vertices 0, 1, and 2.
- **Triangle `t2`** is formed by vertices 2, 3, and 0.

The corresponding `nvertices` list is:

```python
nvertices = [0,1,2, 2,3,0]
```

#### Triangle Edges

Each triangle has three edges:

- **Triangle `t1`** has edges: 0-1, 1-2, and 2-0.
- **Triangle `t2`** has edges: 2-3, 3-0, and 0-2.

The `tedges` list for these edges is:

```python
tedges = [0,1, 1,2, 2,0, 2,3, 3,0, 0,2]
```



Image rendering
----------------------

To render an image, run the following command:

```bash
python renderAnImage.py xcamPos yCamPos zCamPos imgName
```

Here, `xcamPos`, `yCamPos`, and `zCamPos` specify the camera position from which the image will be rendered, and `imgName` specifies the name of the output image.

**Example:**

```bash
python renderAnImage.py 1 5 120 abc
```

To create a high-resolution image, adjust the parameters in the `camera.py` file.

Compute Interval Image
--------------------------

Define the region constraints in the `environment.py` file by setting the `initCubeCon` variable. This variable defines the region constraint in 3D space using three variables for each dimension: `xp0` for the x-axis, `yp0` for the y-axis, and `zp0` for the z-axis. An example constraint is provided below:

**Example Constraint:**

```python
initCubeCon = And(10*xp0>=1, 1000*xp0<=101, 10*yp0>=45, 1000*yp0<=4501, 10*zp0>=1215, 1000*zp0<=121501)
```

> **Note**: Make sure that all coefficients are real numbers and avoid division operators in the constraint.

To compute an interval image, run:

```bash
python intervalImageMain.py
```

The intervals for each pixel in the image are saved in two files:

- `globlaMin.txt` — Contains the lower bounds of intervals for each pixel.
- `globalMax.txt` — Contains the upper bounds of intervals for each pixel.

The code uses the default camera setup specified in `camera.py`.


