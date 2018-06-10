import cv2
import numpy as np
import sys


def Inverse_G_correction(c):
    if(c<0.03928):
        c = float(c)/12.92
    else:
        c = (float(c+0.055)/1.055)**2.4
    return c

def gammacorrection(c):
    if (c<0.00304):
        c = c*12.92
    else:
        c = (1.055*(c**(1/2.4)) - 0.055)
    return c

def convertBRG_to_LUV(bgrimage):
    rows, cols, bands = bgrimage.shape # bands == 3
    luvImage = np.zeros([rows, cols, bands], dtype=np.float)
    min_l = 361
    max_l =0 
    for i in range(0, rows) :
         for j in range(0, cols) :
            b, g, r = bgrimage[i,j]

            
            #divide by 255
            b = float(b)/255.0
            g = float(g)/255.0
            r = float(r)/255.0

            #invgamma correction
            b = Inverse_G_correction(b)
            g = Inverse_G_correction(g)
            r = Inverse_G_correction(r)            

            #linear transformation
            x = 0.412453*r + 0.357580*g + 0.180423*b
            y = 0.212671*r + 0.715160*g + 0.072169*b
            z = 0.019334*r + 0.119193*g + 0.950227*b

            xw, yw, zw = 0.95, 1.0, 1.09
            uw = float(4 * xw)/(xw + 15*yw + 3*zw)
            vw = float(9 * yw)/(xw + 15*yw + 3*zw)

            t = float(y)/yw
            if (t>0.008856):
                l = 116*(t**(1.0/3.0)) - 16
            else:
                l = 903.3*t
            
            d = x + 15*y + 3*z
            if d==0:
                ud = 0
                vd = 0
            else:                
               ud = 4*x/d
               vd = 9*y/d
            u = 13*l*(ud - uw)
            v = 13*l*(vd - vw)
            
            if (l>max_l):
               max_l = l
            if (l<min_l):
               min_l = l
            luvImage[i,j] = [l,u,v]
    
    return luvImage



def Convert_LUV_to_BGR(luvImage):

    rows, cols, bands = luvImage.shape # bands == 3
    bgrImage = np.zeros([rows, cols, bands], dtype=np.float)
    for i in range(0, rows):
         for j in range(0, cols):
            l, u, v = luvImage[i,j]

            xw, yw, zw = 0.95, 1.0, 1.09
            uw = float(4 * xw)/(xw + 15*yw + 3*zw)
            vw = float(9 * yw)/(xw + 15*yw + 3*zw)

            if l==0:
                ud, vd = 0,0
            else:

                ud = float(u + 13*uw*l)/float(13*l)
                vd = float(v + 13*vw*l)/float(13*l)

            if (l>7.9996):
                y = (((l+16)/116)**3)*yw
            else:
                y = (l/903.3)*yw

            if vd ==0:
                x,z = 0,0
            else:
                x = y*2.25*ud/vd
                z = y*(3 - 0.75*ud - 5*vd)/vd

            r = 3.240479*x + (-1.53715)*y + (-0.498535)*z
            g = (-0.969256)*x + 1.875991*y + 0.041556*z
            b = 0.055648*x + (-0.204043)*y +  1.057311*z

            r = gammacorrection(r)
            g = gammacorrection(g)
            b = gammacorrection(b)

            r = round(r*255)
            g = round(g*255)
            b = round(b*255)

            bgrImage[i,j] = [b,g,r]

    return bgrImage






if(len(sys.argv) != 7) :
    print(sys.argv[0], ": takes 6 arguments. Not ", len(sys.argv)-1)
    print("Expecting arguments: w1 h1 w2 h2 ImageIn ImageOut.")
    print("Example:", sys.argv[0], " 0.2 0.1 0.8 0.5 fruits.jpg out.png")
    sys.exit()

w1 = float(sys.argv[1])
h1 = float(sys.argv[2])
w2 = float(sys.argv[3])
h2 = float(sys.argv[4])
name_input = sys.argv[5]
name_output = sys.argv[6]

if(w1<0 or h1<0 or w2<=w1 or h2<=h1 or w2>1 or h2>1) :
    print(" arguments must satisfy 0 <= w1 < w2 <= 1, 0 <= h1 < h2 <= 1")
    sys.exit()

inputImage = cv2.imread(name_input, cv2.IMREAD_COLOR)
if(inputImage is None) :
    print(sys.argv[0], ": Failed to read image from: ", name_input)
    sys.exit()

cv2.imshow("input image: " + name_input, inputImage)

luvImage = convertBRG_to_LUV(inputImage)


rows, cols, bands = luvImage.shape # bands == 3
W1 = int(round(w1*(cols-1)))
H1 = int(round(h1*(rows-1)))
W2 = int(round(w2*(cols-1)))
H2 = int(round(h2*(rows-1)))

max_l = 0
min_l = 361
print (rows, cols, W1, H1, W2, H2)
for i in range(H1, H2) :
    for j in range(W1, W2) :
        #v, u, l= luvImage[i, j]
        l,u,v = luvImage[i, j]
        if (l>max_l):
            max_l = l
        if (l<min_l):
            min_l = l
print (max_l,min_l)

outputluvImage = np.copy(luvImage)

for i in range(0, rows) :
    for j in range(0, cols) :
        l, u, v= luvImage[i, j]
        #l, u, v= luvImage[i, j]
        if (l>max_l):                
            outputluvImage[i][j][0] = 100                
        elif (l<min_l):           
            outputluvImage[i][j][0] = 0
        else:
            outputluvImage[i][j][0] = (l-min_l)*100/(max_l - min_l)


bgroutImage = Convert_LUV_to_BGR(outputluvImage)
#cv2.imshow("output luv", outputImage)
#bgroutImage = cv2.cvtColor(outputImage, cv2.COLOR_Luv2BGR)
cv2.imshow("output:", bgroutImage)
cv2.imwrite(name_output, bgroutImage);


# # wait for key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()