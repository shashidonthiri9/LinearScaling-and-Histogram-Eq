import cv2
import numpy as np
import sys
from collections import Counter

def invgammacorrection(c):
    if(c<0.03928):
        c = float(c)/12.92
    else:
        c = (float(c+0.055)/1.055)**2.4

    return c

def gammacorrectiorn(c):
    if (c<0.00304):
        c = c*12.92
    else:
        c = (1.055*(c**(1/2.4)) - 0.055)

    return c

def convertBgrToLuv(bgrimage):
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
            b = invgammacorrection(b)
            g = invgammacorrection(g)
            r = invgammacorrection(r)            

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



def convertLuvToBgr(luvImage):

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

            r = gammacorrectiorn(r)
            g = gammacorrectiorn(g)
            b = gammacorrectiorn(b)

            r = round(r*255)
            g = round(g*255)
            b = round(b*255)

            bgrImage[i,j] = [b,g,r]

    return bgrImage


def find_min_max_l(luvImage, w1, h1, w2, h2, rows, cols):

    max_l = 0
    min_l = 361

    W1 = int(round(w1*(cols-1)))
    H1 = int(round(h1*(rows-1)))
    W2 = int(round(w2*(cols-1)))
    H2 = int(round(h2*(rows-1)))

    for i in range(H1, H2) :
        for j in range(W1, W2) :
            #v, u, l= luvImage[i, j]
            l,u,v = luvImage[i, j]
            if (l>max_l):
                max_l = l
            if (l<min_l):
                min_l = l

    return max_l, min_l


def find_j_map(tmp,rows,cols):
    
    h = Counter()
    
    for i in range(0,rows):
        for j in range(0, cols):
            
            L,u,v = tmp[i,j]
            
            if(L <= 0):
                L = 0
            elif(L > 100 ):
                L = 100
                
            
            h[(round(L))] += 1
    
    
    f = Counter()
    
#    print(h)
#    print(sum(h.values()))
    
    for key in range(0,101):
        f[key] = h[key] + f[key-1]
    
#    print(f)
    
    j = Counter()
    for key in range(0,101):
        
        val = np.floor(((f[key-1] + f[key])/2.0) * (101.0/f[100]))
        
        if(val > 100 ):
            j[key] = 100
        else:
            j[key] = val
    
    
    return(j)

def hist_equalize(img, rows, cols,min_l, max_l, j_map):
    for i in range(0,rows):
        for j in range(0, cols):
            
            L = img[i,j][0]
            
            if(L < min_l):
                img[i,j][0] = 0.0 
            elif(L >= max_l):
                img[i,j][0] = 100.0
            else:
                img[i,j][0] = j_map[int(round(L))]
                
            if(img[i,j][0] <= 0):
                img[i,j][0] = 0
            elif(img[i,j][0] > 100):
                img[i,j][0] = 100
            
    return(img)

def main():

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

    luvImage = convertBgrToLuv(inputImage)
    rows, cols, bands = luvImage.shape # bands == 3
    
    max_l, min_l = find_min_max_l(luvImage, w1, h1, w2, h2, rows, cols)
    j_map = find_j_map(luvImage, rows, cols)
    scaled_image = hist_equalize(luvImage, rows, cols, min_l, max_l, j_map)

    scaled_bgr_image = convertLuvToBgr(scaled_image)

    cv2.imshow("output:", scaled_bgr_image)
    cv2.imwrite(name_output, scaled_bgr_image);
       
    # # wait for key to exit
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()