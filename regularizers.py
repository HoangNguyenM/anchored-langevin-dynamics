import numpy as np

# Define regularizers
# x has shape (batch_size, sample_size, d), output has shape (batch_size, sample_size)
def get_regularizer(reg_name, *, coef = 1, a = 10):
    def Lasso(x):
        return np.sum(np.abs(x), axis = -1) * coef

    SCAD_coef_1 = a*coef
    SCAD_coef_2 = coef**2 * (a+1)/2
    def SCAD(x):
        # case 1: x <= coef
        # case 2: coef < x <= a*coef
        # case 3 : a*coef < x
        abs_x = np.abs(x)

        matrix1 = abs_x * coef
        matrix2 = (2*SCAD_coef_1*abs_x - abs_x**2 - coef**2)/(2*a-2)
        matrix3 = np.full(abs_x.shape, SCAD_coef_2)

        label3 = np.maximum(np.sign(abs_x-SCAD_coef_1), np.full(abs_x.shape, 0))
        label2 = np.maximum(np.sign(abs_x-coef), np.full(abs_x.shape, 0))
        label2 = label2 - label3
        label1 = np.full(abs_x.shape,1) - label3 - label2
        
        result = matrix1 * label1 + matrix2 * label2 + matrix3 * label3

        return np.sum(result, axis = -1)

    MCP_coef_1 = a*coef
    MCP_coef_2 = coef**2*a/2
    def MCP(x):
        # case 1: x <= a*coef
        # case 2: a*coef < x
        abs_x = np.abs(x)

        matrix1 = coef*abs_x - abs_x**2/(2*a)
        matrix2 = np.full(abs_x.shape, MCP_coef_2)

        label2 = np.maximum(np.sign(abs_x-MCP_coef_1), np.full(abs_x.shape, 0))
        label1 = np.full(abs_x.shape,1) - label2
        
        result = matrix1 * label1 + matrix2 * label2

        return np.sum(result, axis = -1)
    
    if reg_name == 'Lasso':
        return Lasso
    elif reg_name == 'SCAD':
        return SCAD
    elif reg_name == 'MCP':
        return MCP
    else:
        raise NotImplementedError(f"Regularizer name {reg_name} not implemented")
    
# Calculate the smoothed version of regularizers
# x has shape (sample_size, d), output has shape (sample_size,)
def get_ref_regularizer(reg_name, *, coef = 1, a = 10, varep = 0.5):
    def Lasso(x):
        return np.sum(np.sqrt(np.square(x)+varep**2), axis = -1) * coef
    
    SCAD_coef_1 = a*coef
    SCAD_coef_2 = 2*coef*(a**2*coef**2+varep**2)**0.5
    SCAD_coef_3 = coef*(coef**2+2*varep**2)
    SCAD_coef_4 = 2*(a**2*coef**2+varep**2)**0.5-2*(coef**2+varep**2)**0.5
    SCAD_coef_5 = coef**3*(a**2-1)/SCAD_coef_4
    def SCAD(x):
        # case 1: x <= coef
        # case 2: coef < x <= a*coef
        # case 3 : a*coef < x

        abs_x = np.abs(x)
        smoothed_x = np.sqrt(np.square(abs_x) + varep**2)

        matrix1 = smoothed_x * coef
        matrix2 = (SCAD_coef_2 * smoothed_x - coef * np.square(abs_x) - SCAD_coef_3)/SCAD_coef_4
        matrix3 = np.full(abs_x.shape, SCAD_coef_5)

        label3 = np.maximum(np.sign(abs_x-SCAD_coef_1), np.full(abs_x.shape, 0))
        label2 = np.maximum(np.sign(abs_x-coef), np.full(abs_x.shape, 0))
        label2 = label2 - label3
        label1 = np.full(abs_x.shape,1) - label3 - label2
        
        result = matrix1 * label1 + matrix2 * label2 + matrix3 * label3

        return np.sum(result, axis = -1)

    MCP_coef_1 = a*coef
    MCP_coef_2 = 2*(a**2*coef**2+varep**2)**0.5
    MCP_coef_3 = coef*(a**2*coef**2+2*varep**2)/MCP_coef_2
    def MCP(x):
        # case 1: x <= a*coef
        # case 2: a*coef < x
        abs_x = np.abs(x)
        smoothed_x = np.sqrt(np.square(abs_x) + varep**2)

        matrix1 = coef * smoothed_x - coef * np.square(abs_x)/MCP_coef_2
        matrix2 = np.full(abs_x.shape, MCP_coef_3)

        label2 = np.maximum(np.sign(abs_x-MCP_coef_1), np.full(abs_x.shape, 0))
        label1 = np.full(abs_x.shape,1) - label2
        
        result = matrix1 * label1 + matrix2 * label2

        return np.sum(result, axis = -1)
    
    if reg_name == 'Lasso':
        return Lasso
    elif reg_name == 'SCAD':
        return SCAD
    elif reg_name == 'MCP':
        return MCP
    else:
        raise NotImplementedError(f"Regularizer name {reg_name} not implemented")
    
# Calculate the gradients of the smoothed version of regularizers
# The gradient is also added by L2 gradient due to mixed regularizer
# x has shape (sample_size, d), output has shape (sample_size, d)
def get_grad_regularizer(reg_name, *, coef = 1, a = 10, varep = 0.5, l2_coef = 0.5):
    def Lasso(x):
        return (x / np.sqrt(np.square(x)+varep**2)) * coef + 2 * l2_coef * x

    SCAD_coef_1 = a*coef
    SCAD_coef_2 = 2*coef*(a**2*coef**2+varep**2)**0.5
    SCAD_coef_4 = 2*(a**2*coef**2+varep**2)**0.5-2*(coef**2+varep**2)**0.5
    def SCAD(x):
        # case 1: x <= coef
        # case 2: coef < x <= a*coef
        # case 3 : a*coef < x

        abs_x = np.abs(x)
        smoothed_x = np.sqrt(np.square(abs_x) + varep**2)

        matrix1 = (x / smoothed_x) * coef
        matrix2 = (SCAD_coef_2 * (x / smoothed_x) * coef - 2 * coef * x)/SCAD_coef_4
        matrix3 = np.full(abs_x.shape, 0)

        label3 = np.maximum(np.sign(abs_x-SCAD_coef_1), np.full(abs_x.shape, 0))
        label2 = np.maximum(np.sign(abs_x-coef), np.full(abs_x.shape, 0))
        label2 = label2 - label3
        label1 = np.full(abs_x.shape,1) - label3 - label2
        
        result = matrix1 * label1 + matrix2 * label2 + matrix3 * label3

        return result + 2 * l2_coef * x

    MCP_coef_1 = a*coef
    MCP_coef_2 = 2*(a**2*coef**2+varep**2)**0.5
    def MCP(x):
        # case 1: x <= a*coef
        # case 2: a*coef < x

        abs_x = np.abs(x)
        smoothed_x = np.sqrt(np.square(abs_x) + varep**2)

        matrix1 = (x / smoothed_x) * coef - 2* coef * x/MCP_coef_2
        matrix2 = np.full(abs_x.shape, 0)

        label2 = np.maximum(np.sign(abs_x-MCP_coef_1), np.full(abs_x.shape, 0))
        label1 = np.full(abs_x.shape,1) - label2
        
        result = matrix1 * label1 + matrix2 * label2

        return result + 2 * l2_coef * x
    
    if reg_name == 'Lasso':
        return Lasso
    elif reg_name == 'SCAD':
        return SCAD
    elif reg_name == 'MCP':
        return MCP
    else:
        raise NotImplementedError(f"Regularizer name {reg_name} not implemented")