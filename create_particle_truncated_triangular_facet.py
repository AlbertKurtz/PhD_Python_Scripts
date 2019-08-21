import numpy
from scipy.spatial import ConvexHull

# Alberto Flor 2019
# Creates a truncated cube nanoparticle with FCC structure with given
# lattice parameter.
# takes a large cube with initial edge given by INITIAL_CUBE_EDGE and then finds the
# points generating the convex hull for the desired truncation.
# saves two files, one .xyz and one .data good as LAMMPS starting configuration.

#To change the code for other structures, like BCC, it should be sufficient to change the
# definition of "primitive_cell" variable in the main function

UNITS = "Angstrom"
INITIAL_CUBE_EDGE = 20
TRUNCATION_VALUE = 1. # from 0 to 1
LATTICE_PARAMETER= 3.89
ELEMENT = "Pd"



def allthesymmetry(vector):
    symmetries = numpy.array([[1, 1, 1], [1, 1, -1], [1, -1, -1], [-1, -1, -1],
                              [1, -1, 1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1]])

    return list(vector * a for a in symmetries)


def gen_hullpoints(edge, percentage):
    # half edge
    he = 0.5 * edge
    first_point = numpy.array([he * (1 - percentage), he * (1 - percentage), he])
    second_point = numpy.array([he * (1 - percentage), he, he * (1 - percentage)])
    third_point = numpy.array([he, he * (1 - percentage), he * (1 - percentage)])

    points = numpy.concatenate((allthesymmetry(first_point), allthesymmetry(second_point), allthesymmetry(third_point)))
    return points


def find_extrema(list_of_points):
    min_x = 1e10
    max_x = -1e10
    min_y = 1e10
    max_y = -1e10
    min_z = 1e10
    max_z = -1e10
    for atom in list_of_points:
        min_x = min(min_x, atom[0])
        max_x = max(max_x, atom[0])
        min_y = min(min_y, atom[1])
        max_y = max(max_y, atom[1])
        min_z = min(min_z, atom[2])
        max_z = max(max_z, atom[2])
    print([[min_x, max_x], [min_y, max_y], [min_z, max_z]])
    return [[min_x, max_x], [min_y, max_y], [min_z, max_z]]


def in_extrema(atom, extrema):
    return (atom[0] >= extrema[0][0] and atom[0] < extrema[0][1] and
            atom[1] >= extrema[1][0] and atom[1] < extrema[1][1] and
            atom[2] >= extrema[2][0] and atom[2] < extrema[2][1])


def point_in_hull_1(hull, pnt):
    '''
    Checks if `pnt` is inside the convex hull.
    `hull` -- a QHull ConvexHull object
    `pnt` -- point array of shape (3,)
    '''
    new_hull = ConvexHull(numpy.concatenate((hull.points, [pnt])))
    if numpy.array_equal(new_hull.vertices, hull.vertices):
        return True
    return False


def find_extreme_length(hull_points):
    n_points = len(hull_points)
    extreme_length = 0
    for index in range(0, n_points):
        for index2 in range(index + 1, n_points):
            extreme_length = max(extreme_length, numpy.linalg.norm(hull_points[index] - hull_points[index2]))

    return extreme_length


def point_in_shape(atom, points):
    extreme_length = find_extreme_length(points)
    is_true = True
    for a in points:
        for b in points:
            if (numpy.linalg.norm(atom - a) + numpy.linalg.norm(atom - b) <= extreme_length):
                is_true *= True
            else:
                is_true *= False
    return is_true


def max_radius(hull, points):
    min_length = 1e10
    for facet in hull.simplices:
        three_points = [points[int(facet[0])], points[facet[1]], points[facet[2]]]
        length = 0
        for pt in three_points:
            length += pt / 3
        min_length = min(min_length, numpy.linalg.norm(length))
    return min_length / 2

def baricenter(points):
    origin = numpy.zeros(3)
    for point in points:
        origin+= point
    return origin/len(points)

def are_inside_hull(hull, points):
    hullpoints = hull.points
    equations = hull.equations
    origin = baricenter(hullpoints)
    value = True
    for point in points:
        for equation in equations:
            if sum(point*equation[:-1])> equation[-1]:
                value *= True
            else:
                value *= False
    return value

def origin_in_or_out(equations, point):
    value = True
    for equation in equations:
        if sum(point * equation[:-1]) < equation[-1]:
            value *= True
        else:
            value *= False
    if value == True:
        return 'less'
    else:
        return 'greater'

def is_inside_hull(equations, point, less_or_greater):
    value = True
    if less_or_greater == 'less':
        for equation in equations:
            if sum(point*equation[:-1])< equation[-1]:
                value *=True
            else:
                value *=False
        return value
    else:
        for equation in equations:
            if sum(point*equation[:-1])> equation[-1]:
                value *=True
            else:
                value *=False
        return value

def main():
    #defines parameters
    units = UNITS
    initial_cube_edge = INITIAL_CUBE_EDGE
    truncation_percentage = TRUNCATION_VALUE
    lattice_param = LATTICE_PARAMETER
    element = ELEMENT
    n_cells = int(0.5*initial_cube_edge/lattice_param)

    #calls function generating the points for the convex hull
    hullpoints = gen_hullpoints(initial_cube_edge, truncation_percentage)

    hull = ConvexHull(hullpoints)
    print("surface = {} {}^2\n volume = {} {}^3".format(hull.area, units, hull.volume, units))

    equations = hull.equations

    # finds the baricenter of the particle in order to define the direction of the vector normal to the surface
    # so to be able to distinguish atoms inside or outside the surfaces defining the convex hull
    origin = baricenter(hullpoints)
    less_or_greater = origin_in_or_out(equations, origin)

    # this is the primitive cell... by changing it with, for example,
    # primitive_cell = numpy.array([lattice_param*numpy.array([0., 0. ,0.]), lattice_param *numpy.array([.5, .5, .5])])
    # we have a BCC structure.
    primitive_cell = numpy.array([lattice_param*numpy.array([0., 0. ,0.]),
                                  lattice_param *numpy.array([.5, .5, 0.]),
                                  lattice_param *numpy.array([0., .5, .5]),
                                  lattice_param *numpy.array([.5, 0., .5])])

    atoms = []
    # replicates the atoms in the primitive cells along all 6 directions and checks whether new atoms are inside
    # the convex hull. If yes, new atoms are stored in the list "atoms"
    
    for i in range(-n_cells, n_cells+1):
        for j in range(-n_cells, n_cells+1):
            for k in range(-n_cells, n_cells+1):
                for atom in primitive_cell:
                    point = atom + numpy.array([i*lattice_param,
                                                j*lattice_param,
                                                k*lattice_param])
                    if is_inside_hull(equations, point, less_or_greater):
                        atoms.append(point)
    print("n atoms = {}\n".format(len(atoms)))

    filename = "ideal_{}A_{}_{}trunc".format(initial_cube_edge,element,int(truncation_percentage*100))

    with open(filename +".xyz", 'w') as ofile:
        ofile.write("{}\n\n".format(len(atoms)))
        for line in atoms:
            ofile.write("{}\t{}\t{}\t{}\n".format(element, line[0], line[1], line[2]))

    with open(filename +".data", 'w') as ofile:
        ofile.write("\n{} atoms\n".format(len(atoms)))
        ofile.write("1 atom types\n")
        ofile.write("-{} {} xlo xhi\n".format(initial_cube_edge*2, initial_cube_edge*2))
        ofile.write("-{} {} ylo yhi\n".format(initial_cube_edge * 2, initial_cube_edge * 2))
        ofile.write("-{} {} zlo zhi\n".format(initial_cube_edge * 2, initial_cube_edge * 2))
        ofile.write("\nAtoms\n\n")
        index = 1
        for line in atoms:
            ofile.write("{}\t1\t{}\t{}\t{}\n".format(index, line[0], line[1], line[2]))
            index += 1

main()
