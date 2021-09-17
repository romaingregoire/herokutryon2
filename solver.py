import queue
import math
import cvxpy as cp
import numpy

PRECISION_CONSTANT = 10 ** (-8)
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
    def __init__(self, symbols=[], constraints=[], values=[], eq_result_parent=numpy.Inf, eq_result=0,
                 objective_function=None, left_child=None,
                 right_child=None):
        self.symbols = symbols
        self.constraints = constraints
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

    def __get_constraints__(self):
        return self.constraints

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

    def __set_constraints__(self, constraints):
        self.constraints = constraints

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


def get_closest(value, symbols, index_variable):
    left, right = numpy.Inf, numpy.Inf
    if symbols[index_variable][1] is None:
        """case without no constraints to add to this value"""
        return None, None
    elif len(symbols[index_variable][1]) > 0:
        """case where the user is passing a vector in parameter"""
        for vector_values in symbols[index_variable][1]:
            if vector_values < 0:
                raise ValueError("This subexpression you are trying to create in not dgp")
            if abs(value - left) > abs(value - vector_values) and vector_values < value:
                left = vector_values
            if abs(value - right) > abs(vector_values - value) and vector_values > value:
                right = vector_values
    elif len(symbols[index_variable][1] == 0):
        """case with general constraints +1 -1"""
        left = math.floor(value)
        right = math.ceil(value)
    return left, right


def new_constraints(current_node, chosen):
    """First of all we need to create the new contraints according to the choosen variable. To choose which variable we
    are going to use we just take them in order x -> y -> z"""
    if chosen >= len(current_node.__get_symbols__()):
        return None, None
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
    new_right_constraints = right_closest_value <= right_copy_symbols[chosen][0]

    """Now that we have created the new contraints for left and right, which will always be greater on the right and 
    lower on the left"""
    left_constraints = [constraint for constraint in current_node.__get_constraints__()]
    left_constraints.append(new_left_constraints)
    right_constraints = [constraint for constraint in current_node.__get_constraints__()]
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
    left_problem = cp.Problem(left_obj, left_node.__get_constraints__())
    left_eq_res = left_problem.solve(gp=True)
    left_values = [symbol[0].value for symbol in left_copy_symbols]
    left_node.__set_eq_result__(left_eq_res)
    left_node.__set_values__(left_values)
    left_node.__set_symbols__(left_copy_symbols)

    right_obj = cp.Minimize(right_node.__get_objective_function__())
    right_problem = cp.Problem(right_obj, right_node.__get_constraints__())
    right_eq_res = right_problem.solve(gp=True)
    right_values = [symbol[0].value for symbol in right_copy_symbols]
    right_node.__set_eq_result__(right_eq_res)
    right_node.__set_values__(right_values)
    right_node.__set_symbols__(left_copy_symbols)

    return left_node, right_node


def respect_property(current_node):
    for i in range(0, len(current_node.__get_symbols__())):
        symbol, vector_associated = current_node.__get_symbols__()[i]
        if vector_associated is None:
            continue
        elif len(vector_associated) == 0:
            associated_value = current_node.__get_values__()[i]
            floored = math.floor(associated_value)
            ceiled = math.ceil(associated_value)
            closest = 0
            if abs(floored - associated_value) < abs(associated_value - ceiled):
                closest = floored
            else:
                closest = ceiled
            if abs(associated_value - closest) < abs(floored - ceiled) * PRECISION_CONSTANT:
                continue
            else:
                return i
        else:
            current_value = current_node.__get_values__()[i]
            lower_bound, upper_bound = get_closest(current_value, current_node.__get_symbols__(), i)
            closest = 0
            if abs(current_value - lower_bound) < abs(current_value - upper_bound):
                closest = lower_bound
            else:
                closest = upper_bound

            if abs(closest - current_value) < abs(lower_bound - upper_bound) * PRECISION_CONSTANT:
                continue
            else:
                return i
    return len(current_node.__get_symbols__())


def corrected_node(current_node, index_variable):
    associated_value = current_node.__get_values__()[index_variable]
    symbol, vector_associated = current_node.__get_symbols__()[index_variable]
    potential_symbols = [symbol[0] for symbol in current_node.__get_symbols__()]
    if vector_associated is None:
        return current_node
    elif 0 <= len(vector_associated):
        closest = 0
        if len(vector_associated) == 0:
            floored = math.floor(associated_value)
            ceiled = math.ceil(associated_value)
            if abs(floored - associated_value) < abs(associated_value - ceiled):
                closest = floored
            else:
                closest = ceiled
        else:
            lower_bound, upper_bound = get_closest(associated_value, current_node.__get_symbols__(), index_variable)
            closest = 0
            if abs(associated_value - lower_bound) < abs(associated_value - upper_bound):
                closest = lower_bound
            else:
                closest = upper_bound
        added_constraints = [closest == current_node.__get_symbols__()[index_variable][0]]
        for constraint in current_node.__get_constraints__():
            added_constraints.append(constraint)
        obj = cp.Minimize(current_node.__get_objective_function__())
        copy_pb = cp.Problem(obj, added_constraints)
        potential_new_res = copy_pb.solve(gp=True)
        if not (potential_new_res is None) and potential_new_res < current_node.__get_eq_result__():
            current_node.__set_values__([symbol.value for symbol in potential_symbols])
            current_node.__set_eq_result__(potential_new_res)


def branch_and_bound_solve(objective_function, contraints, symbols):
    """Here we create the objective function considering the function passed as parameter for now on we are going to focus
    on minimisation, this could be easily upgraded to both maximisation and minimisation later."""
    obj = cp.Minimize(objective_function)
    problem = cp.Problem(obj, contraints)
    eq_res = problem.solve(gp=True)

    """Here we create the list that store the values of x, y and z gotten from the first optimization"""
    problem_values = [symbol[0].value for symbol in symbols]

    """We then create the root node of our tree and set his differents attributes"""
    root_node = Node()
    root_node.__set_symbols__(symbols)
    root_node.__set_constraints__(contraints)
    root_node.__set_objective_function__(objective_function)
    root_node.__set_values__(problem_values)
    root_node.__set_eq_result__(eq_res)

    """The best solution is going to be the final result of our main function he'll be updated through the function according
    to the branch and bound method"""
    best_solution = [numpy.Inf] * len(symbols)
    best_eq_result = numpy.Inf

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
        print("the value", end=' ')
        for value in current_node.__get_values__():
            print(value, end=' ')
        print("The current value of the equation is: ", current_node.__get_eq_result__(), end=' ')
        print()
        index_variable = respect_property(current_node)
        if index_variable == len(current_node.__get_symbols__()):
            if current_node.__get_eq_result__() < best_eq_result:
                best_eq_result = current_node.__get_eq_result__()
                best_solution = current_node.__get_values__()
            continue
        else:
            corrected_node(current_node, index_variable)
            """We recompute the value of index variable because it's possible that the non discrete value is not the same as the previous one"""
            index_variable = respect_property(current_node)
        left_child, right_child = new_constraints(current_node, index_variable)
        if left_child is None and right_child is None:
            continue
        if not (left_child is None) and not (None in left_child.__get_values__()):
            current_node.__set_left_child__(left_child)
        if not (right_child is None) and not (None in right_child.__get_values__()):
            current_node.__set_right_child__(right_child)
        if not (left_child is None) and not (None in left_child.__get_values__()):
            respect_all_condition = respect_property(left_child)
            if respect_all_condition < len(
                    left_child.__get_symbols__()) and best_eq_result > left_child.__get_eq_result__():
                potential_solutions.put(left_child)
            else:
                if best_eq_result > left_child.__get_eq_result__():
                    best_eq_result = left_child.__get_eq_result__()
                    best_solution = [values for values in left_child.__get_values__()]
        if not (right_child is None) and not (None in right_child.__get_values__()):
            respect_all_condition = respect_property(right_child)
            if respect_all_condition < len(
                    right_child.__get_symbols__()) and best_eq_result > right_child.__get_eq_result__():
                potential_solutions.put(right_child)
            else:
                if best_eq_result > right_child.__get_eq_result__():
                    best_eq_result = right_child.__get_eq_result__()
                    best_solution = [values for values in right_child.__get_values__()]
    return best_solution, best_eq_result


"""Here we declare the variable using the cvxpy library, for now on the only variable usable are the one define bellow"""
x = cp.Variable(pos=True, name="x", )
y = cp.Variable(pos=True, name="y")
z = cp.Variable(pos=True, name="z")
cp_objective_function = z + 35 * (x ** (-1)) + 18.5 * (y ** (-1)) + (y ** 2) + numpy.pi * (z ** (-0.5))
cp_contraints = [x + (y ** 2) + (z ** (1 / 2)) <= 257, 1 * x + 7 * y + 3 * z <= 38]
y_list = []
i = 2
while i <= 19:
    y_list.append(i)
    i += 0.25
cp_symbols = [(x, [1, 4, 7, 12, 18, 36]), (y, y_list), (z, None)]

"""This is the call to the main function"""
final_result_value, final_result = branch_and_bound_solve(cp_objective_function, cp_contraints, cp_symbols)
print("x value is: ", final_result_value[0], "y value is: ",
      final_result_value[1],
      "z value is: ", final_result_value[2],
      "the final result of the equation is: ", final_result)
print()
print("////////////////////////////////////////////////////")
print("////////////////////////////////////////////////////")
print("///////second test with less variable///////////////")
print("////////////////////////////////////////////////////")
print("////////////////////////////////////////////////////")
print()
less_variable_function = 35 * (x ** (-1)) + 18.5 * (y ** (-1)) + (y ** 2) + numpy.pi
less_variable_contraints = [x + (y ** 2) <= 257, 1 * x + 7 * y + 3 <= 38]
less_variable_symbols = [(x, [1, 4, 7, 12, 18]), (y, [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9])]
final_less_variable_value, final_less_variable_result = branch_and_bound_solve(less_variable_function,
                                                                               less_variable_contraints,
                                                                               less_variable_symbols)
print("x value is: ", final_less_variable_value[0], "y value is: ",
      final_less_variable_value[1],
      "the final result of the equation is: ", final_less_variable_result)

print()
print("////////////////////////////////////////////////////")
print("////////////////////////////////////////////////////")
print("////////third test with less variable///////////////")
print("////////////////////////////////////////////////////")
print("////////////////////////////////////////////////////")
print()

test_function = x
test_contraints = [x == 1, x >= 1]
test_obj = cp.Minimize(test_function)
test_pb = cp.Problem(test_obj, test_contraints)
test_solve = test_pb.solve(gp=True)
print("The result of double constraint with same variable is: ", test_solve)
