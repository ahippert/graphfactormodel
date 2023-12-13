from typing import Any
from numbers import Real




## Could not use abc metaclass because need to instanciate in __add__, __sub__, etc.
class OptimizationFunction():
    """
    Base class for optimization functions.
    They should have at least two methods. One to compute the cost,
    the other to compute the Euclidean gradient.

    Beyond providing a template, this base class provides __add__,
    __sub__, __neg__, __mul__ and __rmul__ methods to ease operations
    on cost functions.
    """
    def __init__(self) -> None:
        pass

    def __call__(self, point: Any, *args, **kwds) -> float:
        """
        Cost function.

        Parameters
        ----------
        point : Any
            point for which the cost function is evaluated.

        Returns
        -------
        float
            cost function at point.
        """
        return self.cost(point)

    def cost(self, point: Any, *args, **kwds) -> float:
        """
        Cost function.

        Parameters
        ----------
        point : Any
            point for which the cost function is evaluated.

        Returns
        -------
        float
            cost function at point.
        """
        pass

    def euclidean_gradient(self, point: Any, *args, **kwds) -> Any:
        """
        Euclidean gradient.

        Parameters
        ----------
        point : Any
            point for which the Euclidean gradient is evaluated.

        Returns
        -------
        Any
            Euclidean gradient at point
        """
        pass

    def compose(self, mapping_between_manifolds):
        """
        Composition of optimization function with a mapping between manifolds.
        A new optimization function on the arrival manifold is obtained.

        Parameters
        ----------
        mapping_between_manifolds : MappingBetweenManifolds
            mapping from one manifold onto another.

        Returns
        -------
        OptimizationFunction
            optimization function defined on the arrival manifold of mapping_between_manifolds.
        """
        res = OptimizationFunction()
        def cost_new(*x): return self.cost(mapping_between_manifolds.mapping(*x))
        def euclidean_gradient_new(*x): return mapping_between_manifolds.differential_adjoint(*x,self.euclidean_gradient(mapping_between_manifolds.mapping(*x)))
        res.cost = cost_new
        res.euclidean_gradient = euclidean_gradient_new
        return res

    def __add__(self, other):
        """
        Addition of OptimizationFunction.

        Parameters
        ----------
        other : OptimizationFunction or Real
            either another optimization function or a scalar to be added.

        Returns
        -------
        OptimizationFunction
            resulting optimization function. If other is OptimizationFunction,
            costs and Euclidean gradients of self and other are added. If other
            is a scalar, then it is added to the cost of self and the gradient
            remains unchanged.
        """
        res = OptimizationFunction()
        if isinstance(other, OptimizationFunction):
            def cost_new(*x): return self.cost(*x) + other.cost(*x)
            def euclidean_gradient_new(*x): return self.euclidean_gradient(*x) + other.euclidean_gradient(*x)
        elif isinstance(other, Real): # This is quite useless in practice... Keep ?
            def cost_new(*x): return self.cost(*x) + other
            def euclidean_gradient_new(*x): return self.euclidean_gradient(*x)
        else:
            raise ValueError('OptimizationFunction.__add__: type not handled, other should be either OptimizationFunction or Real')
        res.cost = cost_new
        res.euclidean_gradient = euclidean_gradient_new
        return res

    def __sub__(self, other):
        """
        Subtraction of OptimizationFunction.

        Parameters
        ----------
        other : OptimizationFunction or Real
            either another optimization function or a scalar to be subtracted.

        Returns
        -------
        OptimizationFunction
            resulting optimization function. If other is OptimizationFunction,
            costs and Euclidean gradients of self and other are subtracted. If other
            is a scalar, then it is subtracted to the cost of self and the gradient
            remains unchanged.
        """
        res = OptimizationFunction()
        if isinstance(other, OptimizationFunction):
            def cost_new(*x): return self.cost(*x) - other.cost(*x)
            def euclidean_gradient_new(*x): return self.euclidean_gradient(*x) - other.euclidean_gradient(*x)
        elif isinstance(other,Real): # This is quite useless in practice... Keep ?
            def cost_new(*x): return self.cost(*x) - other
            def euclidean_gradient_new(*x): return self.euclidean_gradient(*x)
        else:
            raise ValueError('OptimizationFunction.__sub__: type not handled, other should be either OptimizationFunction or Real')
        res.cost = cost_new
        res.euclidean_gradient = euclidean_gradient_new
        return res
    
    def __neg__(self):
        """
        Negative of OptimizationFunction.

        Returns
        -------
        OptimizationFunction
            Opposite of the orignal optimization function.
        """
        res = OptimizationFunction()
        def cost_new(*x): return -self.cost(*x)
        def euclidean_gradient_new(*x): return -self.euclidean_gradient(*x)
        res.cost = cost_new
        res.euclidean_gradient = euclidean_gradient_new
        return res
    
    def __mul__(self,other):
        """
        Multiplication of OptimizationFunction.
        Only handles multiplication by scalars.

        Parameters
        ----------
        other : Real
            scalar value.

        Returns
        -------
        OptimizationFunction
            original optimization function weighted by scalar other.
        """
        res = OptimizationFunction()
        if isinstance(other, Real):
            def cost_new(*x): return other * self.cost(*x)
            def euclidean_gradient_new(*x): return other * self.euclidean_gradient(*x)
        else:
            raise ValueError('OptimizationFunction.__sub__: type not handled, for now other should Real')
        res.cost = cost_new
        res.euclidean_gradient = euclidean_gradient_new
        return res
        
    __rmul__ = __mul__




# should be abstract class probably -> YES
class MappingBetweenManifolds():
    """
    Abstract class with the template for mappings between two manifolds.
    """
    def __init__(self) -> None:
        pass

    def mapping(self, point):
        """
        Mapping from one manifold onto the other.

        Parameters
        ----------
        point : Any
            point on the starting manifold.
        
        Returns
        -------
        Any
            point on the arrival manifold.
        """
        pass

    def differential(self, point, tangent_vector):
        """
        Differential of the mapping between the two manifolds.

        Parameters
        ----------
        point : Any
            point on the starting manifold.
        tangent_vector : Any
            tangent vector at point.
        
        Returns
        -------
        Any
            tangent vector at mapping(point).
        """
        pass

    def differential_adjoint(self, point, tangent_vector_arrival): # Euclidean metrics
        """
        Adjoint of the differential of the mapping between the two manifolds.
        It is expected to be defined through the Euclidean metrics on both manifolds.

        Parameters
        ----------
        point : Any
            point on the starting manifold.
        tangent_vector_arrival : Any
            tangent vector at mapping(point) on the arrival manifold.

        Returns
        -------
        Any
            tangent vector at point on the starting manifold.
        """
        pass


# # Shouldn't it be directly a method of OptimizationFunction class ? -> Done
# class ComposedOptimizationFunction(OptimizationFunction):
#     """
    
#     """
#     def __init__(self, optimization_function, mapping_between_manifolds) -> None:
#         self.optimization_function = optimization_function
#         self.mapping_between_manifolds = mapping_between_manifolds

#     def cost(self, point) -> float:
#         return self.optimization_function.cost(self.mapping_between_manifolds.mapping(point))
    
#     def euclidean_gradient(self, point) -> Any:
#         return self.mapping_between_manifolds.differential_adjoint(point,self.optimization_function.euclidean_gradient(self.mapping_between_manifolds.mapping(point)))