# src/solvers.py

import sympy
from sympy import symbols, Function, Eq, dsolve, sympify, lambdify
import numpy as np
from scipy.integrate import odeint

def solve_dynamic_ode(order, highest_deriv_expr_str, t_points, y0_values):
    
    if len(y0_values) != order:
        raise ValueError(f"Se esperaban {order} condiciones iniciales, pero se recibieron {len(y0_values)}")

    t = symbols('t')
    y_func = Function('y')
    
    locals_map = {'t': t, 'y_val': y_func(t)}
    deriv_names = ['y_p'] + [f"y_{'p'*(i+2)}" for i in range(order - 1)]
    
    symbolic_derivs = []
    for i in range(order):
        deriv = y_func(t).diff(t, i + 1)
        symbolic_derivs.append(deriv)
        if i < len(deriv_names):
            locals_map[deriv_names[i]] = deriv
    
    try:
        symbolic_expr = sympify(highest_deriv_expr_str, locals=locals_map)
    except Exception as e:
        print(f"Error al parsear la expresión simbólica: {e}")
        return None, None, f"Error de sintaxis en EDO: {e}"

    solution_symbolic_obj = None
    solution_symbolic_str = "No se encontró solución simbólica."
    try:
        highest_deriv = y_func(t).diff(t, order)
        edo_eq = Eq(highest_deriv, symbolic_expr)
        solution_symbolic_obj = dsolve(edo_eq, y_func(t))
        solution_symbolic_str = str(solution_symbolic_obj)
    except Exception as e:
        print(f"Info: No se pudo encontrar una solución simbólica. Error: {e}")

    y_syms = symbols(f'y_0:{order}')
    lambda_map = {y_func(t).diff(t, i): y_syms[i] for i in range(order)}
    lambda_map[y_func(t)] = y_syms[0]
    
    numeric_expr = symbolic_expr.subs(lambda_map)
    f_num = lambdify((t, *y_syms), numeric_expr, 'numpy')

    def ode_system(Y, t):
        dY_dt = np.zeros_like(Y)
        dY_dt[0:order-1] = Y[1:order]
        dY_dt[order-1] = f_num(t, *Y)
        return dY_dt

    solution_numeric = odeint(ode_system, y0_values, t_points)
    
    return solution_numeric, solution_symbolic_str