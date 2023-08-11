from skimage.exposure import adjust_gamma
from matplotlib import pyplot as plt
from PIL import Image
import pytesseract
import cv2
import numpy as np
import os


path = r"C:\Program Files\Tesseract-OCR"
pytesseract.pytesseract.tesseract_cmd = path + r"\tesseract.exe"

cwd = os.getcwd()
img_dir = '\\trabalho\img' + input("Digite o número da imagem:\n") + '.png'
img = Image.open(cwd + img_dir) 
img.save(".\\trabalho\\new_img.png")
img = cv2.imread(cwd+"\\trabalho\\new_img.png", cv2.IMREAD_GRAYSCALE)
img= cv2.resize(img, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
cv2.imwrite(".\\trabalho\\new_img.png", img)
#cv2.imshow("Imagem Inicial", img)
#cv2.waitKey()

def thresholding(type):
    img = cv2.imread(".\\trabalho\\new_img.png", 0) 
    limiarVal = int(input("Valor Limiar:\n"))
    maxVal = int(input("Valor Maximo:\n"))
    match type:
        case 0:
            #Binário
            ret, img = cv2.threshold(img, limiarVal, maxVal, cv2.THRESH_BINARY)
        case 1:
            #Binário Invertido
            ret, img = cv2.threshold(img, limiarVal, maxVal, cv2.THRESH_BINARY_INV)
        case 2:
            #Thresholding Adaptativo Médio
            img = cv2.adaptiveThreshold(img, maxVal, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
        case 3:
            #Thresholding Adaptativo Gaussiano
            img = cv2.adaptiveThreshold(img, maxVal, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2)
    cv2.imwrite(".\\trabalho\\new_img.png", img)

def smoothing(type):
    img = cv2.imread(".\\trabalho\\new_img.png", 0)
    ordem_kernel = int(input("Ordem do kernel (Deve ser positivo e ímpar)\n"))
    match type:
        case 0:
            #2D Convolution
            kernel = np.ones((ordem_kernel, ordem_kernel), np.float32)/(ordem_kernel*ordem_kernel)
            img = cv2.filter2D (img, -1, kernel)
        case 1:
            #Averaging
            img = cv2.blur(img,(ordem_kernel,ordem_kernel))
        case 2:
            #Gaussian Blurring
            img = cv2.GaussianBlur(img, (ordem_kernel, ordem_kernel), cv2.BORDER_DEFAULT)
        case 3:
            #Median Blurring
            img = cv2.medianBlur(img, ordem_kernel)
    cv2.imwrite(".\\trabalho\\new_img.png", img)

def morphological(type):
    img = cv2.imread(".\\trabalho\\new_img.png", 0)
    ordem_kernel = int(input("Ordem do kernel (Deve ser positivo e ímpar)\n"))
    kernel = np.ones((ordem_kernel,ordem_kernel),np.uint8)
    match type:
        case 0:
            #Erosion
            num_iterations = int(input("Número de Iterações\n"))
            img = cv2.erode(img, kernel, iterations = num_iterations)
        case 1:
            #Dilation
            num_iterations = int(input("Número de Iterações\n"))
            img = cv2.dilate(img, kernel, iterations = num_iterations)
        case 2:
            #Opening
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        case 3:
            #Closing
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(".\\trabalho\\new_img.png", img)

def gradients(type):
    img = cv2.imread(".\\trabalho\\new_img.png", 0)
    match type:
        case 0:
            #Sobel
            ordem_kernel = int(input("Ordem do kernel (Deve ser 1, 3, 5 ou 7)\n"))
            dtype = int(input("\t\t>>> ESCOLHA:\n\t\t0: Vertical\n\t\t1: Horizontal\n"))
            match dtype:
                case 0:
                    x = 1
                    y = 0
                case 1:
                    x = 0
                    y = 1
            img = cv2.Sobel(img, cv2.CV_64F, x, y, ksize=ordem_kernel)
        case 1:
            #Laplacian
            img = cv2.Laplacian(img, cv2.CV_64F)
    cv2.imwrite(".\\trabalho\\new_img.png", img)

def canny():
    minVal = int(input("Valor minino:\n"))
    maxVal = int(input("Valor maximo:\n"))
    img = cv2.imread(".\\trabalho\\new_img.png", 0)
    img = cv2.Canny (img, minVal, maxVal)
    cv2.imwrite(".\\trabalho\\new_img.png", img)

def histograma():
    img = cv2.imread(".\\trabalho\\new_img.png")
    plt.hist(img.ravel(),256,[0,256])
    plt.show()

    
while True:
    print("0: EXECUTAR LEITURA\n1: Threshold\n2: Suavização\n3: Morfologia\n4: Detecção de Contornos\n5: Canny\n6: Histograma\n7: Correção de Gamma\n8: Rollback")
    tratamento = int(input("ESCOLHA\n"))

    match tratamento:
        case 0:
            break
        case 1:
            #Thresholding
            type = int(input("\t>>> ESCOLHA:\n\t0: Binário\n\t1: Binário Invertido\n\t2: Adaptativo Médio\n\t3: Adaptativo Gaussiano\n"))
            thresholding(type)
        case 2:
            #Smoothing 
            type = int(input("\t>>> ESCOLHA:\n\t0: Convolução 2D\n\t1: Blur\n\t2: Blur Gaussiano\n\t3: Blur Mediano\n"))
            smoothing(type)
        case 3:
            #Morphological Transformations 
            type = int(input("\t>>> ESCOLHA:\n\t0: Erosão\n\t1: Dilatação\n\t2: Abertura\n\t3: Fechamento\n"))
            morphological(type)
        case 4:
            #Image Gradients
            type = int(input("\t>>> ESCOLHA:\n\t0: Sobel\n\t1: Laplacian\n"))
            gradients(type)
        case 5:
            #Canny
            canny()
        case 6:
            #Histograma
            histograma()
        case 7:
            #Gamma correction
            gamma = float(input("Valor gamma:\n"))
            img = cv2.imread(".\\trabalho\\new_img.png", 0)
            gamma_corrected = adjust_gamma(img, gamma)
            cv2.imwrite(".\\trabalho\\new_img.png", img)
        case 8:
            #Rollback
            img = Image.open(cwd + img_dir)
            img.save(".\\trabalho\\new_img.png")
            img = cv2.imread(".\\trabalho\\new_img.png", cv2.IMREAD_GRAYSCALE)
            img= cv2.resize(img, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(".\\trabalho\\new_img.png", img)
    img = cv2.imread(cwd+"\\trabalho\\new_img.png")
    #cv2.imshow("Resultado Tratamento", img)
    #cv2.waitKey()        

text = pytesseract.image_to_string(img, lang="eng")
print(text)