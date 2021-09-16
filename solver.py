import queue
import math
import cvxpy as cp
import numpy

"""This the class that we are going to use through the entire process it allows us to store the different result of the node but also 
to access the values at different point in time if needed. The class is a simple binary tree that store different information:
-The symbols of the equation x, y, z and so on
-The contraints of the equation
-The values of our symbols gotten from the resolution of our equation
-The result of the computation of the equation known as eq_result
-The objective function which could be gotten from minimisation or maximisation of the function passed in entry by the user
-The left and right child if their respective value are more optimize
-"""


class Node:
    def __init__(self, symbols=[], contraints=[], values=[], eq_result_parent=numpy.Inf, eq_result=0,
                 objective_function=None, left_child=None,
                 right_child=None):
        self.symbols = symbols
        self.contraints = contraints
        self.values = values
        self.eq_result_parent = eq_result_parent
        self.eq_result = eq_result
        self.objective_function = objective_function
        self.left_child = left_child
        self.right_child = right_child

    """Here we define the getters and setters that are methods that allows us to avoid the direct acces to the value of our data structure.
    This is used very often in object oriented programming."""

    def __get_symbols__(self):
        return self.symbols

    def __get_contraints__(self):
        return self.contraints

    def __get_values__(self):
        return self.values

    def __get_eq_result_parent__(self):
        return self.eq_result_parent

    def __get_eq_result__(self):
        return self.eq_result

    def __get_objective_function__(self):
        return self.objective_function

    def __get_weight__(self):
        return self.weight

    def __get_left_child__(self):
        return self.left_child

    def __get_right_child__(self):
        return self.right_child

    def __set_symbols__(self, symbols):
        self.symbols = symbols

    def __set_contraints__(self, contraints):
        self.contraints = contraints

    def __set_values__(self, values):
        self.values = values

    def __set_eq_result_parent__(self, eq_result_parent):
        self.eq_result_parent = eq_result_parent

    def __set_eq_result__(self, eq_result):
        self.eq_result = eq_result

    def __set_objective_function__(self, objective_function):
        self.objective_function = objective_function

    def __set_left_child__(self, left_child):
        self.left_child = left_child

    def __set_right_child__(self, right_child):
        self.right_child = right_child


""" def best_parameter(self):
        best_parameter = 0
        difference = numpy.Inf
        if self.symbols[0].value - math.floor(self.symbols[0].value) < math.ceil(self.symbols[0].value) - self.symbols[0].value:
            difference = self.symbols[0].value - math.floor(self.symbols[0].value)
        else:
            difference = math.ceil(self.symbols[0].value) - self.symbols[0].value
        for i in range(1, len(self.symbols)):
            floored = self.symbols[i].value - math.floor(self.symbols[i].value)
            ceiled = math.ceil(self.symbols[i].value) - self.symbols[i].value
            if floored < ceiled:
                if difference > floored:
                    best_parameter = i
                    difference = floored
            else:
                if difference > ceiled:
                    best_parameter = i
                    difference = floored
        return best_parameter"""


def get_closest(value, symbols, chosen):
    left, right = -numpy.Inf, numpy.Inf
    if symbols[chosen][1] is None:
        """case without no constraints to add to this value"""
        return None, None
    elif len(symbols[chosen][1]) > 0:
        """case where the user is passing a vector in parameter"""
        for vector_values in symbols[chosen][1]:
            if vector_values < 0:
                raise ValueError("This subexpression you are trying to create in not dgp")
            if (value - left) > (value - vector_values):
                left = vector_values
            if (right - value) > (vector_values - value):
                right = vector_values
    elif len(symbols[chosen][1] == 0):
        """case with general constraints +1 -1"""
        if math.floor(value) - 1 < 0:
            left = None
        right = math.ceil(value)
    return left, right


def new_constraints(current_node, chosen):
    """First of all we need to create the new contraints according to the choosen variable. To choose which variable we
    are going to use we just take them in order x -> y -> z"""
    symbols = current_node.__get_symbols__()
    left_copy_symbols = [symbol for symbol in symbols]
    right_copy_symbols = [symbol for symbol in symbols]
    left_closest_value, right_closest_value = get_closest(current_node.__get_values__()[chosen], symbols, chosen)
    if left_closest_value is None and right_closest_value is None:
        return None, None
    """current_node.__get_values__()[chosen]"""
    new_left_constraints = None
    if not (left_closest_value is None):
        new_left_constraints = left_closest_value >= left_copy_symbols[chosen][0]
    new_right_constraints = math.ceil(current_node.__get_values__()[chosen]) <= right_copy_symbols[chosen][0]

    """Now that we have created the new contraints for left and right, which will always be greater on the right and 
    lower on the left"""
    left_constraints = [constraint for constraint in current_node.__get_contraints__()]
    left_constraints.append(new_left_constraints)
    right_constraints = [constraint for constraint in current_node.__get_contraints__()]
    right_constraints.append(new_right_constraints)

    """Then we create the new node, we are just missing the result of their respective resolution which we are gonna add bellow"""
    left_node = Node(left_copy_symbols, left_constraints, current_node.__get_values__(),
                     current_node.__get_eq_result_parent__(),
                     0,
                     current_node.__get_objective_function__(), None, None)
    right_node = Node(right_copy_symbols, right_constraints, current_node.__get_values__(),
                      current_node.__get_eq_result_parent__(),
                      0,
                      current_node.__get_objective_function__(), None, None)

    """Here we compute the left and right equation, we add the respective result to the nodes and we return them"""
    left_obj = cp.Minimize(left_node.__get_objective_function__())
    left_problem = cp.Problem(left_obj, left_node.__get_contraints__())
    left_eq_res = left_problem.solve(gp=True)
    left_values = [symbol[0].value for symbol in left_copy_symbols]
    left_node.__set_eq_result__(left_eq_res)
    left_node.__set_values__(left_values)
    left_node.__set_symbols__(left_copy_symbols)

    right_obj = cp.Minimize(right_node.__get_objective_function__())
    right_problem = cp.Problem(right_obj, right_node.__get_contraints__())
    right_eq_res = right_problem.solve(gp=True)
    right_values = [symbol[0].value for symbol in right_copy_symbols]
    right_node.__set_eq_result__(right_eq_res)
    right_node.__set_values__(right_values)
    right_node.__set_symbols__(left_copy_symbols)

    return left_node, right_node


def branch_and_bound_solve(objective_function, contraints, symbols):
    """Here we create the objective function considering the function passed as parameter for now on we are going to focus
    on minimisation, this could be easily upgraded to both maximisation and minimisation later."""
    obj = cp.Minimize(objective_function)
    problem = cp.Problem(obj, contraints)
    eq_res = problem.solve(gp=True)

    """This is just a tool to help us debug the code"""
    print("x value is: ", symbols[0][0].value, "y value is: ", symbols[1][0].value, "z value is: ", symbols[2][0].value,
          "the result of the equation is: ", eq_res)

    """Here we create the list that store the values of x, y and z gotten from the first optimization"""
    problem_values = [symbols[0][0].value, symbols[1][0].value, symbols[2][0].value]

    """We then create the root node of our tree and set his differents attributes"""
    root_node = Node()
    root_node.__set_symbols__(symbols)
    root_node.__set_contraints__(contraints)
    root_node.__set_objective_function__(objective_function)
    root_node.__set_values__(problem_values)
    root_node.__set_eq_result__(eq_res)

    """The best solution is going to be the final result of our main function he'll be updated through the function according
    to the branch and bound method"""
    best_solution = root_node

    """We are using a queue as the limitation of our loop, because it's arguiably one of the best data structure to represent 
    a tree."""
    potential_solutions = queue.Queue()
    potential_solutions.put(root_node)

    """Index variable keep track of which variable x, y or z we are optimizing during the branch and bound, for example 
    if we dont have any better solution when index variable is equal to 0  we increment it so that we can do the branch 
    and bound over y and so on."""
    index_variable = 0
    while not (potential_solutions.empty()):
        current_node = potential_solutions.get()
        left_child, right_child = new_constraints(current_node, index_variable)
        if left_child is None and right_child is None:
            index_variable += 1
            if index_variable >= len(symbols):
                break
        current_node.__set_right_child__(right_child)
        if not (left_child is None):
            current_node.__set_left_child__(left_child)
        if not (left_child is None) and (left_child.__get_eq_result__() < best_solution.__get_eq_result__()
                                         or right_child.__get_eq_result__() < best_solution.__get_eq_result__()):
            if not (left_child is None) and left_child.__get_eq_result__() < best_solution.__get_eq_result__():
                best_solution = left_child
            if right_child.__get_eq_result__() < best_solution.__get_eq_result__():
                best_solution = right_child
            if not (left_child is None) and left_child.__get_eq_result__() < best_solution.__get_eq_result__() + (
                    0.1 * best_solution.__get_eq_result__()):
                potential_solutions.put(left_child)
            if right_child.__get_eq_result__() < best_solution.__get_eq_result__() + (
                    0.1 * best_solution.__get_eq_result__()):
                potential_solutions.put(right_child)
        elif index_variable < len(symbols):
            index_variable += 1
        """This is just a tool to help us debug the code"""
        print("x value is: ", current_node.__get_values__()[0], "y value is: ", current_node.__get_values__()[1],
              "z value is: ", current_node.__get_values__()[2],
              "the result of the equation is: ", current_node.__get_eq_result__(), "for an index value of: ",
              index_variable)
    return best_solution


"""Here we declare the variable using the cvxpy library, for now on the only variable usable are the one define bellow"""
x = cp.Variable(pos=True, name="x", )
y = cp.Variable(pos=True, name="y")
z = cp.Variable(pos=True, name="z")
cp_objective_function = z + 35 * (x ** (-1)) + 18.5 * (y ** (-1)) + (y ** 2) + numpy.pi * (z ** (-0.5))
"""cp_contraints = [x + (y ** 2) + (z ** (1 / 2)) <= 257, 1 * x + 7 * y + 3 * z <= 38, x <= 2.0]
obj = cp.Minimize(cp_objective_function)
problem = cp.Problem(obj, cp_contraints)
res = problem.solve(solver=cp.ECOS, reltol_inacc=10 ** (-1), gp=True)
print("x value is: ", x.value, "y value is: ", y.value,
      "z value is: ", z.value,
      "the result of the equation is: ", res)"""

cp_contraints = [x + (y ** 2) + (z ** (1 / 2)) <= 257, 1 * x + 7 * y + 3 * z <= 38]
cp_symbols = [(x, [1, 6, 7, 11, 13, 14, 23, 22, 53, 36]), (y, [3, 89, 90, 25, 12, 1, 23, 42, 36, 9]), (z, [])]

"""This is the call to the main function"""
final_result = branch_and_bound_solve(cp_objective_function, cp_contraints, cp_symbols)
print("x value is: ", final_result.__get_symbols__()[0][0].value, "y value is: ",
      final_result.__get_symbols__()[1][0].value,
      "z value is: ", final_result.__get_symbols__()[2][0].value,
      "the result of the equation is: ", final_result.__get_eq_result__())
