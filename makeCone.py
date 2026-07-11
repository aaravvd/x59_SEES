import math

# ============================================================
# Geometry
# ============================================================

length = 1.0                    # cone length (m)
halfAngleDeg = 10.0             # half-angle
halfAngle = math.radians(halfAngleDeg)

radius = length * math.tan(halfAngle)

n = 200                         # number of circumferential divisions


# ============================================================
# Helper functions
# ============================================================

def normal(v1, v2, v3):
    ax = v2[0] - v1[0]
    ay = v2[1] - v1[1]
    az = v2[2] - v1[2]

    bx = v3[0] - v1[0]
    by = v3[1] - v1[1]
    bz = v3[2] - v1[2]

    nx = ay*bz - az*by
    ny = az*bx - ax*bz
    nz = ax*by - ay*bx

    mag = math.sqrt(nx*nx + ny*ny + nz*nz)

    if mag == 0:
        return (0.0, 0.0, 0.0)

    return (nx/mag, ny/mag, nz/mag)


def writeFacet(f, p1, p2, p3):
    nrm = normal(p1, p2, p3)

    f.write(
        f"facet normal {nrm[0]:.12e} {nrm[1]:.12e} {nrm[2]:.12e}\n"
    )
    f.write("outer loop\n")
    f.write(f"vertex {p1[0]:.12e} {p1[1]:.12e} {p1[2]:.12e}\n")
    f.write(f"vertex {p2[0]:.12e} {p2[1]:.12e} {p2[2]:.12e}\n")
    f.write(f"vertex {p3[0]:.12e} {p3[1]:.12e} {p3[2]:.12e}\n")
    f.write("endloop\n")
    f.write("endfacet\n")


# ============================================================
# Generate vertices
# ============================================================

tip = (0.0, 0.0, 0.0)
baseCenter = (length, 0.0, 0.0)

circle = []

for i in range(n):
    theta = 2.0 * math.pi * i / n

    y = radius * math.cos(theta)
    z = radius * math.sin(theta)

    circle.append((length, y, z))


# ============================================================
# Write STL
# ============================================================

with open("cone.stl", "w") as f:

    f.write("solid cone\n")

    # --------------------------------------------------------
    # Side surface
    #
    # Ordering chosen so normals point outward.
    # --------------------------------------------------------

    for i in range(n):

        p1 = tip
        p2 = circle[(i + 1) % n]
        p3 = circle[i]

        writeFacet(f, p1, p2, p3)

    # --------------------------------------------------------
    # Base disk
    #
    # Viewed from downstream (+x), normal points +x.
    # --------------------------------------------------------

    for i in range(n):

        p1 = baseCenter
        p2 = circle[i]
        p3 = circle[(i + 1) % n]

        writeFacet(f, p1, p2, p3)

    f.write("endsolid cone\n")

print("Generated cone.stl")
print(f"Length      : {length}")
print(f"Half-angle  : {halfAngleDeg} deg")
print(f"Base radius : {radius:.6f} m")
print(f"Triangles   : {2*n}")
