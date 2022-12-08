# Jerrett Martin
# Utilities for representing functions and calculating their Radon transform

import numpy as np
import matplotlib.pyplot as plt



######## MISC. UTILITIES ########

SQRT_2 = np.sqrt(2)

class Vec2:
    '''Custom barebones Vec2 class'''

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __add__(self, rhs):
        '''Add two Vec2's'''

        return Vec2(self.x + rhs.x, self.y + rhs.y)
    
    def __sub__(self, rhs):
        '''Subtract two Vec2's'''

        return Vec2(self.x - rhs.x, self.y - rhs.y)
    
    def __mul__(self, scalar):
        '''Scalar Multiplication'''

        return Vec2(self.x * scalar, self.y * scalar)
    __rmul__ = __mul__

    def perpendicular(self):
        '''Returns a vector with the same magnitude perpendicular to self in the positive direction (+90°)'''

        return Vec2(-self.y, self.x)



######## SUBFUNCTIONS CORE ########

class Subfunction:
    '''An object made up of a condition (lambda) with two arguments and a value (float).
        Represents a subfunction of a 2D function space.
    '''

    def __init__(self, condition, value: float):
        '''condition (lambda x, y), value (float)'''

        self.condition = condition
        self.value = value
    
    def get(self, x: float, y: float):
        '''Returns the value of the function at (x,y) if the condition is met'''

        return (self.value if self.condition(x, y) else None)

class Union(Subfunction):
    '''A union of subfunctions is, itself, a subfunction'''

    def __init__(self, subfunctions, value: float):
        '''subfunctions (array of subfunctions), value (float)'''

        self.subfunctions = subfunctions
        self.value = value
    
    def get(self, x: float, y: float):
        '''returns value if any subfunction in the Union is defined at (x,y)'''

        for sub in self.subfunctions:
            if sub.get(x, y) is not None:
                return self.value
        return None

class Intersection(Subfunction):
    '''An intersection of subfunctions is, itself, a subfunction'''

    def __init__(self, subfunctions, value: float):
        '''subfunctions (array of subfunctions), value (float)'''
        self.subfunctions = subfunctions
        self.value = value

    def get(self, x: float, y: float):
        '''returns value iff every subfunction in the Intersection is defined at (x,y)'''

        for sub in self.subfunctions:
            if sub.get(x, y) is None:
                return None
        return self.value



######## 2D PRIMITIVES ########
# A set of common 2D shapes for use as subfunctions
# All 2D primitives are represented as implicit functions (inequalities) of x and y

class Circle(Subfunction):
    '''The simplest 2D primitive, a circle at (posX, posY) with radius r.'''

    def __init__(self, posX: float, posY: float, r: float, value: float = 0):
        
        self.condition = lambda x,y: (x-posX)**2 + (y-posY)**2 <= (r)**2
        self.value = value

class Ellipse(Subfunction):
    '''An ellipse centered at (posX, posY) with axes of w/2 and h/2, respectively, and rotation of theta'''

    def __init__(self, posX: float, posY: float, w: float, h: float, theta: float, value: float = 0):
        
        self.condition = lambda x,y: pow(((x-posX) * np.cos(theta) + (y-posY) * np.sin(theta)) / w, 2) + pow(((y-posY) * np.cos(theta) - (x-posX) * np.sin(theta)) / h, 2) <= 1
        self.value = value

class Rect(Subfunction):
    '''A rectangle centered at (posX, posY) with sides w and h and rotation of theta'''

    def __init__(self, posX=0, posY=0, w=1, h=1, theta=0, value = 0):

        self.condition = lambda x,y: abs((((x-posX) * np.cos(theta) + (y-posY) * np.sin(theta)) / w) + (((y-posY) * np.cos(theta) - (x-posX) * np.sin(theta)) / h)) + abs((((y-posY) * np.cos(theta) - (x-posX) * np.sin(theta)) / h) - (((x-posX) * np.cos(theta) + (y-posY) * np.sin(theta)) / w)) <= 1
        self.value = value



######## FUNCTION SPACE ########
# A function space is a collection of subfunctions with additional operations

class FunctionSpace:
    '''A collection of subfunctions with additional operations, such as:

        > lintegral(theta, dist, step):
            - Performs a numerical line integral over a line characterized by an angle (theta)
              and a distance from the origin (dist) with configurable step size (step)

        > radon(theta, num, step):
            - Calculates the Radon transform of the subfunction at a given angle (theta),
              where (num) is the number of samples along the line and (step) is the step size
              of lintegral

        > showRadon(theta):
            - Shows a pretty picture of the FunctionSpace and its Radon transform at a given angle (theta)
        
        > show():
            - Shows a pretty picture of the FunctionSpace
    '''

    def __init__(self, subfunc_list = []):
        '''subfunc_list is a list of subfunctions'''

        self.subfunctions = subfunc_list
    
    def include(self, subfunc: Subfunction) -> None:
        '''Includes a new subfunction in the FunctionSpace'''

        self.subfunctions.append(subfunc)
    
    def get(self, x: float, y: float) -> float:
        '''Get the value of the FunctionSpace at (x,y).
            The value at (x,y) is the sum of all subfunctions defined at (x,y)
            If no subfunctions are defined at (x,y), returns 0.
        '''

        valSum = 0

        for subfunc in self.subfunctions:
            val = subfunc.get(x, y)

            # If function is defined here, return the value
            if val is not None:
                valSum += val
        
        return valSum
    
    def lintegral(self, theta=0, dist=0, step=0.005):
        '''Calculates the Radon transform of the subfunction at a given angle (theta),
            where (num) is the number of samples along the line and (step) is the step size
            of lintegral
        '''

        dir = Vec2(np.cos(theta), np.sin(theta))
        omega = dir * dist
        orthomega = dir.perpendicular() # Perpendicular to omega

        sum = 0
        for t in np.arange(-SQRT_2, SQRT_2, step=step):
            point = omega + (orthomega * t)
            
            # Function is not supported outside of [-1,1]x[-1,1]
            if (-1 <= point.x <= 1) and (-1 <= point.y <= 1):
                sum += self.get(point.x, point.y)
        
        sum *= step
        return sum
    
    def radon(self, theta=0, num=50, step=0.001):
        '''Calculates the Radon transform of the subfunction at a given angle (theta),
            where (num) is the number of samples along the line and (step) is the step size
            of lintegral

            Returns a list of values
        '''

        output = []
        for t in np.linspace(-1, 1, num, endpoint=True):
            output.append(self.lintegral(theta, t, step))
        return output
    
    def showRadon(self, theta=0, resolution=256):

        # plt.style.use('_mpl-gallery-nogrid')

        X, Y = np.meshgrid(np.linspace(-1, 1, resolution), np.linspace(-1, 1, resolution))
        Z = []
        for x in range(resolution):
            row = []
            for y in range(resolution):
                row.append(self.get(X[x,y], Y[x,y]))
            Z.append(row)
        
        levels = np.linspace(np.min(Z), np.max(Z), len(self.subfunctions) + 2)

        _, (fx, rx) = plt.subplots(1, 2)
        # fx.contourf(X, Y, Z, levels=levels)

        fx.imshow(Z, origin='lower', extent=[-1.0,1.0,-1.0,1.0], aspect='equal', cmap='gray')

        radon_res = 128
        # rx.set_ylim([0, 1])
        rx.plot(np.linspace(-1, 1, radon_res), self.radon(theta, radon_res))

        plt.show()
    
    def show(self, resolution = 256):

        # plt.style.use('_mpl-gallery-nogrid')

        X, Y = np.meshgrid(np.linspace(-1, 1, resolution), np.linspace(-1, 1, resolution))
        Z = []
        
        for x in range(resolution):
            row = []
            for y in range(resolution):
                row.append(self.get(X[x,y], Y[x,y]))
            Z.append(row)
        
        _, ax = plt.subplots()
        
        ax.imshow(Z, origin='lower', extent=[-1.0,1.0,-1.0,1.0], aspect='equal')
        # ax.imshow(Z, origin='lower', extent=[-1.0,1.0,-1.0,1.0], aspect='equal', cmap='gray')

        plt.show()
    
    def exportRadon(self, output='radon.csv', numAngles=90, numSamples=50):
        with open(output, 'w') as f:
            # Calculate radon data from 0 to 180 degrees, with a spacing of 2 degrees
            for theta in np.linspace(0, np.pi, numAngles):
                rowRadon = self.radon(theta, num=numSamples, step=0.01)
                for i in range(len(rowRadon) - 1):
                    f.write(str(round(rowRadon[i], 5)))
                    f.write(',')
                f.write(str(round(rowRadon[-1], 5)))
                f.write('\n')



######## TEST PHANTOMS ########

class SheppLogan(FunctionSpace):
    '''https://en.wikipedia.org/wiki/Shepp%E2%80%93Logan_phantom
       A representation of the Logan-Shepp phantom, used to approximate the shape of the human head.
    '''

    def __init__(self):
        '''subfunc_list is a list of subfunctions'''

        pi = np.pi

        a = Ellipse(    0,       0,   0.69,  0.92,        0,     2)
        b = Ellipse(    0,       0,  0.345,  0.46,        0,     2)
        b = Ellipse(    0, -0.0184, 0.6624, 0.874,        0, -0.98)
        c = Ellipse( 0.22,       0,   0.11,  0.31, -pi / 10, -0.2)
        d = Ellipse(-0.22,       0,   0.16,  0.41,  pi / 10, -0.2)
        e = Ellipse(    0,    0.35,   0.21,  0.25,        0,  0.1)
        f = Ellipse(    0,     0.1,  0.046, 0.046,        0,  0.1)
        g = Ellipse(    0,    -0.1,  0.046, 0.046,        0,  0.1)
        h = Ellipse(-0.08,  -0.605,  0.046, 0.023,        0,  0.1)
        i = Ellipse(    0,  -0.605,  0.023, 0.023,        0,  0.1)
        j = Ellipse( 0.06,  -0.605,  0.025, 0.046,        0,  0.1)

        self.subfunctions = [a,b,c,d,e,f,g,h,i,j]
        
class Smile(FunctionSpace):
    '''A smile! :)
    '''

    def __init__(self):
        lEye = Ellipse(-0.27, 0.43, 0.125, 0.3, 0, 1)
        rEye = Ellipse( 0.27, 0.43, 0.125, 0.3, 0, 1)

        nose = Rect(0, -0.05, 0.1, 0.1, np.pi/4, 0.75)

        mouth1 = Circle(-0.44,  -0.23, 0.1, 0.5)
        mouth2 = Circle(-0.25, -0.43, 0.1, 0.4)
        mouth3 = Circle(    0,  -0.53, 0.1, 0.3)
        mouth4 = Circle( 0.25, -0.43, 0.1, 0.4)
        mouth5 = Circle( 0.44,  -0.23, 0.1, 0.5)

        self.subfunctions = [lEye, rEye, nose, mouth1, mouth2, mouth3, mouth4, mouth5]
