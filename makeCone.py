import math

length = 1.0
halfAngle = math.radians(10)
radius = length * math.tan(halfAngle)

n = 100

with open("cone.stl", "w") as f:
    f.write("solid cone\n")

    # Base
    for i in range(n):
        a1 = 2*math.pi*i/n
        a2 = 2*math.pi*(i+1)/n

        x1 = radius*math.cos(a1)
        y1 = radius*math.sin(a1)

        x2 = radius*math.cos(a2)
        y2 = radius*math.sin(a2)

        f.write("facet normal 0 0 -1\n")
        f.write("outer loop\n")
        f.write(f"vertex 0 0 {length}\n")
        f.write(f"vertex {x2} {y2} 0\n")
        f.write(f"vertex {x1} {y1} 0\n")
        f.write("endloop\n")
        f.write("endfacet\n")

    # Side
    for i in range(n):
        a1 = 2*math.pi*i/n
        a2 = 2*math.pi*(i+1)/n

        x1 = radius*math.cos(a1)
        y1 = radius*math.sin(a1)

        x2 = radius*math.cos(a2)
        y2 = radius*math.sin(a2)

        f.write("facet normal 0 0 0\n")
        f.write("outer loop\n")
        f.write("vertex 0 0 0\n")
        f.write(f"vertex {x1} {y1} {length}\n")
        f.write(f"vertex {x2} {y2} {length}\n")
        f.write("endloop\n")
        f.write("endfacet\n")

    f.write("endsolid cone\n")
