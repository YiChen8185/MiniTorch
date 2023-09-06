from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    val0, val1 = list(vals), list(vals)
    val0[arg] += epsilon
    val1[arg] -= epsilon
    f_prime = (f(*val0) - f(*val1)) / (2 * epsilon)
    return f_prime
    # raise NotImplementedError("Need to implement for Task 1.1")


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    mark = set()
    ans = []

    def visit(variable: Variable) -> None:
        # check if marked
        if variable.unique_id in mark or variable.is_constant():
            return
        # visit all the parents of current node
        for node in variable.parents:
            visit(node)
        # mark current node visited
        mark.add(variable.unique_id)
        # add current node to ans list
        ans.append(variable)

    visit(variable)
    ans.reverse()
    return ans


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # Task 1.4.
    queue = topological_sort(variable)
    derivative = {}
    derivative[variable.unique_id] = deriv
    for var in queue:
        deriv = derivative[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(deriv)
        else:
            for v, d in var.chain_rule(deriv):
                if v.is_constant():
                    continue
                derivative.setdefault(v.unique_id, 0.0)
                derivative[v.unique_id] = derivative[v.unique_id] + d
    # END Task1.4

    # # Step 0: Call topological sort.
    # top_order = topological_sort(variable)
    # # Step 1: Create dic for Scalalrs and current derivaties
    # dic = {}
    # dic[variable.unique_id] = deriv
    # # Step 2:
    # for var in top_order:
    #     if var.is_leaf():
    #         var.accumulate_derivative(dic[var.unique_id])
    #     else:
    #         for (v, d) in var.chain_rule(dic[var.unique_id]):
    #             if v.is_constant():
    #                 continue
    #             if v.unique_id in dic:
    #                 dic[v.unique_id] += d
    #             else:
    #                 dic[v.unique_id] = d


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
