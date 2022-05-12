
import numpy as np
import math
import matplotlib.pyplot as plt
import time
from numpy.linalg import multi_dot
import matplotlib.pyplot as plt
import pandas as pd
import sys

E = 1
I = 1
A = 1
t0 = time.time()


cerceve_cubuklar = []
yukler = []

mesnet_cokme_listesi = []
kafes_cubuk_durum_raporu = []
kafes_cubuk_q = []
kafes_cubuklar = []
uzay_cubuklar = []
uzay_cubuk_durum_raporu = []
uzay_cubuk_q = []

kafes_dugum_listesi = []
uzay_dugum_listesi = []
cerceve_dugum_listesi = []

kafes_dugum_numara_listesi = []
uzay_dugum_numara_listesi = []
cerceve_dugum_numara_listesi = []


kafes_cubuk_numara_listesi = []
uzay_cubuk_numara_listesi = []
cerceve_cubuk_numara_listesi = []

kafes_yakin = []
kafes_uzak = []
cerceve_yakin = []
cerceve_uzak = []
uzay_yakin = []
uzay_uzak = []

kafes_qN = []
uzay_qN = []
kafes_qF = []
uzay_qF = []
uzay_FNx = []
uzay_FNy = []
uzay_FNz = []
uzay_FFx = []
uzay_FFy = []
uzay_FFz = []
cerceve_FNx = []
cerceve_FNy = []
cerceve_FNz = []
cerceve_FFx = []
cerceve_FFy = []
cerceve_FFz = []

kafes_FNx = []
kafes_FNy = []
kafes_FFx = []
kafes_FFy = []

kafes_menset_numara_listesi = []
uzay_menset_numara_listesi = []
cerceve_menset_numara_listesi = []

kafes_mesnetler = []
uzay_mesnetler = []
cerceve_mesnetler = []

kafes_colors = []
cerceve_colors = []
uzay_colors = []

class mesnet_cokmesi():
    def __init__(self ,  df_numarasi ,  cokme_miktari):
        self.dF_numarasi = df_numarasi
        self.cokme_miktari = cokme_miktari

class yuk():
    def __init__(self ,  dF_numarasi ,  F):
        self.dF_numarasi = dF_numarasi
        self.F = F

class uzay_dugum_noktasi():
    def __init__(self ,  numara ,  x ,  y , z ,   d1x ,  d2y ,  d3z):
        self.numara = numara
        self.x = x
        self.y = y
        self.z = z
        self.d1x = d1x
        self.d2y = d2y
        self.d3z = d3z

        uzay_dugum_listesi.append(self)
        uzay_dugum_numara_listesi.append(self.numara)

class uzay_mesnet():
    def __init__(self ,  numara ,  x ,  y , z ,   d1x ,  d2y ,  d3z):
        self.numara = numara
        self.x = x
        self.y = y
        self.z = z
        self.d1x = d1x
        self.d2y = d2y
        self.d3z = d3z
        uzay_mesnetler.append(self)
        uzay_dugum_listesi.append(self)
        uzay_dugum_numara_listesi.append(self.numara)

class cerceve_dugum_noktasi():
    def __init__(self ,  numara ,  x ,  y ,  d1x ,  d2y ,  d3z):
        self.numara = numara
        self.x = x
        self.y = y
        self.d1x = d1x
        self.d2y = d2y
        self.d3z = d3z
        self.adi = "DN"
        cerceve_dugum_listesi.append(self)
        cerceve_dugum_numara_listesi.append(self.numara)
 
class cerceve_mesnet():
    def __init__(self ,  numara ,  x ,  y ,  d1x ,  d2y ,  d3z):    
        self.numara = numara
        self.x = x
        self.y = y
        self.d1x = d1x
        self.d2y = d2y
        self.d3z = d3z
        self.adi = "M"
        cerceve_mesnetler.append(self)        
        cerceve_dugum_listesi.append(self)
        cerceve_dugum_numara_listesi.append(self.numara)
  

class kafes_dugum_noktasi():
    def __init__(self , numara , x , y , d1x , d2y):
        self.numara = numara
        self.x = x
        self.y = y      
        self.d1x = d1x
        self.d2y = d2y  
        kafes_dugum_listesi.append(self)

        kafes_dugum_numara_listesi.append(self.numara)
        
class kafes_mesnet():
    def __init__(self , numara , x , y , d1x , d2y):
        self.numara = numara
        self.x = x
        self.y = y      
        self.d1x = d1x
        self.d2y = d2y  
        kafes_dugum_listesi.append(self)
        kafes_mesnetler.append(self)
        kafes_dugum_numara_listesi.append(self.numara)
        
class uzay_GLOBAL():
    Q = 0
    D = 0
    K11 = 0
    K21 = 0
    K22 = 0
    K12 = 0
    Q_bilinmeyen = 0
    D_bilinmeyen = 0

    def __init__(self ,  eleman_sayisi ,  dugum_sayisi ,  tutulu_sayisi , mesnet_sayisi):
        self.mesnet_sayisi = mesnet_sayisi
        self.n = dugum_sayisi*3 - mafsal_sayisi
        self.tutulu_sayisi = tutulu_sayisi
        self.tutulu_olmayan = self.n - self.tutulu_sayisi  # tutulu olmayan eleman sayisi
        self.eleman_sayisi = eleman_sayisi
        self.dugum_sayisi = dugum_sayisi
        self.K = np.zeros((self.n ,  self.n))
        self.Q_bilinen = np.zeros((self.tutulu_olmayan ,  1))
        self.D_bilinen = np.zeros((self.tutulu_sayisi ,  1))
        self.Fx = np.zeros(dugum_sayisi)
        self.Fy = np.zeros(dugum_sayisi)
        self.Fz = np.zeros(dugum_sayisi)
        self.Ux = np.zeros(dugum_sayisi)
        self.Uy = np.zeros(dugum_sayisi)
        self.Uz = np.zeros(dugum_sayisi)
        self.UNx = np.zeros(eleman_sayisi)
        self.UNy = np.zeros(eleman_sayisi)
        self.UNz = np.zeros(eleman_sayisi)
        self.UFx = np.zeros(eleman_sayisi)
        self.UFy = np.zeros(eleman_sayisi)
        self.UFz = np.zeros(eleman_sayisi)
        self.Mesnet_Fx = np.zeros(mesnet_sayisi)
        self.Mesnet_Fy = np.zeros(mesnet_sayisi)
        self.Mesnet_Fz = np.zeros(mesnet_sayisi)
        self.Mesnet_Ux = np.zeros(mesnet_sayisi)
        self.Mesnet_Uy = np.zeros(mesnet_sayisi)
        self.Mesnet_Uz = np.zeros(mesnet_sayisi)

class cerceve_GLOBAL():
    Q = 0
    D = 0
    K11 = 0
    K21 = 0
    K22 = 0
    K12 = 0
    Q_bilinmeyen = 0
    D_bilinmeyen = 0

    def __init__(self ,  eleman_sayisi ,  dugum_sayisi ,  tutulu_sayisi , mesnet_sayisi):
        self.mesnet_sayisi = mesnet_sayisi
        self.n = dugum_sayisi*3 - mafsal_sayisi
        self.tutulu_sayisi = tutulu_sayisi
        self.tutulu_olmayan = self.n - self.tutulu_sayisi  # tutulu olmayan eleman sayisi
        self.eleman_sayisi = eleman_sayisi
        self.dugum_sayisi = dugum_sayisi
        self.K = np.zeros((self.n ,  self.n))
        self.Q_bilinen = np.zeros((self.tutulu_olmayan ,  1))
        self.D_bilinen = np.zeros((self.tutulu_sayisi ,  1))
        self.Fx = np.zeros(dugum_sayisi)
        self.Fy = np.zeros(dugum_sayisi)
        self.Fz = np.zeros(dugum_sayisi)
        self.Ux = np.zeros(dugum_sayisi)
        self.Uy = np.zeros(dugum_sayisi)
        self.Uz = np.zeros(dugum_sayisi)
        self.UNx = np.zeros(eleman_sayisi)
        self.UNy = np.zeros(eleman_sayisi)
        self.UNz = np.zeros(eleman_sayisi)
        self.UFx = np.zeros(eleman_sayisi)
        self.UFy = np.zeros(eleman_sayisi)
        self.UFz = np.zeros(eleman_sayisi)
        self.Mesnet_Fx = np.zeros(mesnet_sayisi)
        self.Mesnet_Fy = np.zeros(mesnet_sayisi)
        self.Mesnet_Fz = np.zeros(mesnet_sayisi)
        self.Mesnet_Ux = np.zeros(mesnet_sayisi)
        self.Mesnet_Uy = np.zeros(mesnet_sayisi)
        self.Mesnet_Uz = np.zeros(mesnet_sayisi)

class kafes_GLOBAL():
    Q = 0
    D = 0
    K11 = 0
    K21 = 0
    K22 = 0
    K12 = 0
    Q_bilinmeyen = 0
    D_bilinmeyen = 0
    def __init__(self , eleman_sayisi , dugum_sayisi , tutulu_sayisi , mesnet_sayisi):
        self.mesnet_sayisi = mesnet_sayisi
        self.n = dugum_sayisi*2
        self.tutulu_sayisi = tutulu_sayisi
        self.tutulu_olmayan = self.n - self.tutulu_sayisi #tutulu olmayan eleman sayisi
        self.eleman_sayisi = eleman_sayisi
        self.dugum_sayisi = dugum_sayisi
        self.K = np.zeros((self.n , self.n))
        self.Q_bilinen = np.zeros((self.tutulu_olmayan , 1))
        self.D_bilinen = np.zeros((self.tutulu_sayisi , 1))
        self.Fx = np.zeros(dugum_sayisi)
        self.Fy = np.zeros(dugum_sayisi)
        self.Fz = np.zeros(dugum_sayisi)
        self.Ux = np.zeros(dugum_sayisi)
        self.Uy = np.zeros(dugum_sayisi)
        self.Uz = np.zeros(dugum_sayisi)
        self.UNx = np.zeros(eleman_sayisi)
        self.UNy = np.zeros(eleman_sayisi)
        self.UNz = np.zeros(eleman_sayisi)
        self.UFx = np.zeros(eleman_sayisi)
        self.UFy = np.zeros(eleman_sayisi)
        self.UFz = np.zeros(eleman_sayisi)
        self.Mesnet_Fx = np.zeros(mesnet_sayisi)
        self.Mesnet_Fy = np.zeros(mesnet_sayisi)
        self.Mesnet_Fz = np.zeros(mesnet_sayisi)
        self.Mesnet_Ux = np.zeros(mesnet_sayisi)
        self.Mesnet_Uy = np.zeros(mesnet_sayisi)
        self.Mesnet_Uz = np.zeros(mesnet_sayisi)
class uzay_cubuk_eleman():

    def __init__(self , numara ,  xN ,  yN , zN ,  xF ,  yF , zF ,  dNx ,  dNy ,  dNz ,  dFx ,  dFy ,  dFz ,  E  ,  A):
        self.numara = numara
        self.D = np.zeros((6 ,  1) ,  dtype=float)
        self.Q = np.zeros((6 ,  1) ,  dtype=float)
        self.q = 0
        self.xN = xN
        self.yN = yN
        self.zN = zN
        self.xF = xF
        self.yF = yF
        self.zF = zF
        self.E = E
        self.A = A
        self.L = math.sqrt((self.xF-self.xN)**2 + (self.yF-self.yN)**2 +(self.zF-self.zN)**2  )
        self.dNx = dNx
        self.dNy = dNy
        self.dNz = dNz
        self.dFx = dFx
        self.dFy = dFy
        self.dFz = dFz
        self.qF = 0
        self.qN = 0
        self.tanimla()
        uzay_cubuk_numara_listesi.append(self.numara)

    def tanimla(self):
        self.lambx = (self.xF - self.xN) / self.L
        self.lamby = (self.yF - self.yN) / self.L
        self.lambz = (self.zF - self.zN) / self.L

        self.k_prime = (self.E*self.A/self.L) * np.array([
            [1 , -1],
            [-1 , 1]])

        self.T = np.array([
            [self.lambx ,  self.lamby ,  self.lambz ,  0 ,  0 ,  0] , 
            [0 , 0 , 0 , self.lambx , self.lamby , self.lambz]])

        self.T_T = np.transpose(self.T)
        self.k1 = np.dot(np.dot(self.T_T , self.k_prime),self.T)
        
        self.k = (self.A * self.E /self.L) * np.array ([
            [self.lambx**2 , self.lambx*self.lamby , self.lambx*self.lambz , -self.lambx**2 , -self.lambx*self.lamby , -self.lambx*self.lambz ],
            [self.lamby*self.lambx , self.lamby**2 , self.lamby*self.lambz , -self.lamby*self.lambx , -self.lamby**2 , -self.lamby*self.lambz],
            [self.lambz*self.lambx , self.lambz*self.lamby , self.lambz**2 , -self.lambz*self.lambx , -self.lambz*self.lamby , -self.lambz**2  ],
            [-self.lambx**2 , -self.lambx*self.lamby , -self.lambx*self.lambz , self.lambx**2 , self.lambx*self.lamby ,self.lambx*self.lambz],
            [-self.lamby*self.lambx , -self.lamby**2 , -self.lamby*self.lambz , self.lamby*self.lambx , self.lamby**2 , self.lamby*self.lambz],
            [-self.lambz*self.lambx , -self.lambz*self.lamby , -self.lambz**2 , self.lambz*self.lambx , self.lambz*self.lamby , self.lambz**2]])
        
        
        self.indeks = np.array([
            [(self.dNx-1 ,  self.dNx-1 ,  0 ,  0) ,  (self.dNx-1 ,  self.dNy-1 ,  0 ,  1) ,  (self.dNx-1 ,  self.dNz-1 ,  0 ,  2) , 
             (self.dNx-1 ,  self.dFx-1 ,  0 ,  3) ,  (self.dNx-1 ,  self.dFy-1 ,  0 ,  4) ,  (self.dNx-1 ,  self.dFz-1 ,  0 ,  5)] , 
            [(self.dNy-1 ,  self.dNx-1 ,  1 ,  0) ,  (self.dNy-1 ,  self.dNy-1 ,  1 ,  1) ,  (self.dNy-1 ,  self.dNz-1 ,  1 ,  2) , 
             (self.dNy-1 ,  self.dFx-1 ,  1 ,  3) ,  (self.dNy-1 ,  self.dFy-1 ,  1 ,  4) ,  (self.dNy-1 ,  self.dFz-1 ,  1 ,  5)] , 
            [(self.dNz-1 ,  self.dNx-1 ,  2 ,  0) ,  (self.dNz-1 ,  self.dNy-1 ,  2 ,  1) ,  (self.dNz-1 ,  self.dNz-1 ,  2 ,  2) , 
             (self.dNz-1 ,  self.dFx-1 ,  2 ,  3) ,  (self.dNz-1 ,  self.dFy-1 ,  2 ,  4) ,  (self.dNz-1 ,  self.dFz-1 ,  2 ,  5)] , 
            [(self.dFx-1 ,  self.dNx-1 ,  3 ,  0) ,  (self.dFx-1 ,  self.dNy-1 ,  3 ,  1) ,  (self.dFx-1 ,  self.dNz-1 ,  3 ,  2) , 
             (self.dFx-1 ,  self.dFx-1 ,  3 ,  3) ,  (self.dFx-1 ,  self.dFy-1 ,  3 ,  4) ,  (self.dFx-1 ,  self.dFz-1 ,  3 ,  5)] , 
            [(self.dFy-1 ,  self.dNx-1 ,  4 ,  0) ,  (self.dFy-1 ,  self.dNy-1 ,  4 ,  1) ,  (self.dFy-1 ,  self.dNz-1 ,  4 ,  2) , 
             (self.dFy-1 ,  self.dFx-1 ,  4 ,  3) ,  (self.dFy-1 ,  self.dFy-1 ,  4 ,  4) ,  (self.dFy-1 ,  self.dFz-1 ,  4 ,  5)] , 
            [(self.dFz-1 ,  self.dNx-1 ,  5 ,  0) ,  (self.dFz-1 ,  self.dNy-1 ,  5 ,  1) ,  (self.dFz-1 ,  self.dNz-1 ,  5 ,  2) , 
             (self.dFz-1 ,  self.dFx-1 ,  5 ,  3) ,  (self.dFz-1 ,  self.dFy-1 ,  5 ,  4) ,  (self.dFz-1 ,  self.dFz-1 ,  5 ,  5)]
        ])

        self.D_indeks = np.array([
            [(self.dNx-1 ,  0)] , 
            [(self.dNy-1 ,  1)] , 
            [(self.dNz-1 ,  2)] , 
            [(self.dFx-1 ,  3)] , 
            [(self.dFy-1 ,  4)] , 
            [(self.dFz-1 ,  5)]
        ])

    def qatayici(self ,  D_global , Q_global):
        for i in self.D_indeks:
            for j ,  p in i:
                self.D[p] += D_global[j]
                self.Q[p] += Q_global[j]
        self.q = self.k_prime.dot(self.T).dot(self.D)
        self.qF = self.q[1]
        self.qN = -self.qF

class kafes_cubuk_eleman():
    def __init__(self , numara , xN , yN , xF , yF , dNx , dNy , dFx , dFy , E , A):
        self.numara = numara
        self.D = np.zeros((4,1) , dtype=float)
        self.Q = np.zeros((4,1) , dtype=float)
        self.q = 0
        self.xN = xN
        self.yN = yN
        self.xF = xF
        self.yF = yF
        self.E = E
        self.A = A
        self.L = math.sqrt((self.xF-self.xN)**2 + (self.yF-self.yN)**2 )
        self.dNx = dNx
        self.dNy = dNy
        self.dFx = dFx
        self.dFy = dFy
        self.qF = 0
        self.qN = 0
        self.tanimla()
        kafes_cubuk_numara_listesi.append(self.numara)

        
    def tanimla(self):

        self.lambx = (self.xF - self.xN) / self.L
        self.lamby = (self.yF - self.yN) / self.L

        self.k_prime = (self.A*self.E/self.L) * np.array([
            [1 , -1] , 
            [-1 , 1]
            ])

        self.T = np.array([
            [self.lambx , self.lamby , 0 , 0] , 
            [0 , 0 , self.lambx , self.lamby]
            ])
            
        self.T_T = np.transpose(self.T)
        

        self.k = (self.A* self.E /self.L)* np.array([
            [(self.lambx**2)  ,  (self.lambx*self.lamby)  ,  -(self.lambx**2)  ,  -(self.lambx*self.lamby)] , 
            [(self.lambx*self.lamby)  ,  (self.lamby**2)  ,  -(self.lambx*self.lamby)  ,  -(self.lamby**2)] , 
            [-(self.lambx**2)  ,  -(self.lambx*self.lamby)  ,  (self.lambx**2)  ,  (self.lambx*self.lamby)] , 
            [-(self.lambx*self.lamby)  ,  -(self.lamby**2)  ,  (self.lambx*self.lamby)  ,  (self.lamby**2)]
        ])

        self.indeks = np.array([
            [(self.dNx-1 , self.dNx-1 , 0 , 0)  ,  (self.dNx-1 , self.dNy-1 , 0 , 1)  ,  (self.dNx-1 , self.dFx-1 , 0 , 2)  ,  (self.dNx-1 , self.dFy-1 , 0 , 3)] , 
            [(self.dNy-1 , self.dNx-1 , 1 , 0)  ,  (self.dNy-1 , self.dNy-1 , 1 , 1)  ,  (self.dNy-1 , self.dFx-1 , 1 , 2)  ,  (self.dNy-1 , self.dFy-1 , 1 , 3)] , 
            [(self.dFx-1 , self.dNx-1 , 2 , 0)  ,  (self.dFx-1 , self.dNy-1 , 2 , 1)  ,  (self.dFx-1 , self.dFx-1 , 2 , 2)  ,  (self.dFx-1 , self.dFy-1 , 2 , 3)] , 
            [(self.dFy-1 , self.dNx-1 , 3 , 0)  ,  (self.dFy-1 , self.dNy-1 , 3 , 1)  ,  (self.dFy-1 , self.dFx-1 , 3 , 2)  ,  (self.dFy-1 , self.dFy-1 , 3 , 3)]
        ])
        
        self.D_indeks = np.array([
            [(self.dNx-1  ,  0)] , 
            [(self.dNy-1  ,  1)] , 
            [(self.dFx-1  ,  2)] , 
            [(self.dFy-1  ,  3)]
        ])
        
    def qatayici(self, D_global, Q_global):
        
        for i in self.D_indeks:
            for j  ,   p in i:
                self.D[p] += D_global[j]
                self.Q[p] += Q_global[j]
        self.q = self.k_prime.dot(self.T).dot(self.D)
        self.qF = self.q[1]
        self.qN = -self.qF

class cerceve_cubuk_eleman():

    def __init__(self , numara ,  xN ,  yN ,  xF ,  yF ,  dNx ,  dNy ,  dNz ,  dFx ,  dFy ,  dFz ,  E ,  I ,  A):
        self.numara = numara
        self.D = np.zeros((6 ,  1) ,  dtype=float)
        self.Q = np.zeros((6 ,  1) ,  dtype=float)
        self.q = np.zeros((6 ,  1) ,  dtype=float)
        self.xN = xN
        self.yN = yN
        self.xF = xF
        self.yF = yF
        self.E = E
        self.A = A
        self.I = I
        self.L = math.sqrt((self.xF-self.xN)**2 + (self.yF-self.yN)**2)
        self.dNx = dNx
        self.dNy = dNy
        self.dNz = dNz
        self.dFx = dFx
        self.dFy = dFy
        self.dFz = dFz
        self.tanimla()
        cerceve_cubuk_numara_listesi.append(self.numara)


    def tanimla(self):
        self.lambx = (self.xF - self.xN) / self.L
        self.lamby = (self.yF - self.yN) / self.L

        self.k_prime = np.array([
            [(self.E*self.A/self.L) ,  0 ,  0 ,  (-self.E*self.A/self.L) ,  0 ,  0] , 
            [0 ,  (12*self.E*self.I/self.L**3) ,  (6*self.E*self.I/self.L**2) ,  0 , (-12*self.E*self.I/self.L**3) ,  (6*self.E*self.I/self.L**2)] , 
            [0 ,  (6*self.E*self.I/self.L**2) ,  (4*self.E*self.I/self.L) ,  0 , (-6*self.E*self.I/self.L**2) ,  (2*self.E*self.I/self.L)] , 
            [(-self.E*self.A/self.L) ,  0 ,  0 ,  (self.E*self.A/self.L) ,  0 ,  0] , 
            [0 ,  (-12*self.E*self.I/self.L**3) ,  (-6*self.E*self.I/self.L**2) ,  0 , (12*self.E*self.I/self.L**3) ,  (-6*self.E*self.I/self.L**2)] , 
            [0 ,  (6*self.E*self.I/self.L**2) ,  (2*self.E*self.I/self.L) ,  0 , (-6*self.E*self.I/self.L**2) ,  (4*self.E*self.I/self.L)]
        ])

        self.T = np.array([
            [self.lambx ,  self.lamby ,  0 ,  0 ,  0 ,  0] , 
            [-self.lamby ,  self.lambx ,  0 ,  0 ,  0 ,  0] , 
            [0 ,  0 ,  1 ,  0 ,  0 ,  0] , 
            [0 ,  0 ,  0 ,  self.lambx ,  self.lamby ,  0] , 
            [0 ,  0 ,  0 ,  -self.lamby ,  self.lambx ,  0] , 
            [0 ,  0 ,  0 ,  0 ,  0 ,  1]
        ])

        self.T_T = np.transpose(self.T)
        self.k1 = np.dot(np.dot(self.T_T , self.k_prime),self.T)
        
        
        self.k = np.array([
            [(self.A * self.E/self.L*self.lambx**2 + 12*self.E*self.I/(self.L**3)*self.lamby**2) , ((self.A * self.E/self.L -12*self.E*self.I/(self.L**3))*self.lambx*self.lamby ) , -(6*self.E*self.I/(self.L**2)*self.lamby) , -((self.A*self.E/self.L*self.lambx**2) + (12*self.E*self.I/(self.L**3)*self.lamby**2) ) , -((self.A * self.E/self.L -12*self.E*self.I/(self.L**3))*self.lambx*self.lamby ) , -(6*self.E*self.I/(self.L**2)*self.lamby) ],
            [((self.A * self.E/self.L -12*self.E*self.I/(self.L**3))*self.lambx*self.lamby ) , ((self.A*self.E/self.L*self.lamby**2) + (12*self.E*self.I/(self.L**3)*self.lambx**2)) , (6*self.E*self.I/(self.L**2)*self.lambx) , -((self.A * self.E/self.L -12*self.E*self.I/(self.L**3))*self.lambx*self.lamby ) , -(self.A * self.E/self.L*self.lamby**2 + 12*self.E*self.I/(self.L**3)*self.lambx**2) , (6*self.E*self.I/(self.L**2)*self.lambx) ],
            [-(6*self.E*self.I/(self.L**2)*self.lamby) , (6*self.E*self.I/(self.L**2)*self.lambx) , (4*self.E*self.I/self.L) , (6*self.E*self.I/(self.L**2)*self.lamby) , -((6*self.E*self.I/(self.L**2)*self.lambx)) , (2*self.E*self.I/self.L) ],
            [-(self.A * self.E/self.L*self.lambx**2 + 12*self.E*self.I/(self.L**3)*self.lamby**2) , -((self.A * self.E/self.L -12*self.E*self.I/(self.L**3))*self.lambx*self.lamby ) , (6*self.E*self.I/(self.L**2)*self.lamby) , ((self.A*self.E/self.L*self.lambx**2) + (12*self.E*self.I/(self.L**3)*self.lamby**2) ) , ((self.A * self.E/self.L -12*self.E*self.I/(self.L**3))*self.lambx*self.lamby ) , (6*self.E*self.I/(self.L**2)*self.lamby) ],
            [-((self.A * self.E/self.L -12*self.E*self.I/(self.L**3))*self.lambx*self.lamby ) , -((self.A*self.E/self.L*self.lamby**2) + (12*self.E*self.I/(self.L**3)*self.lambx**2)) , -(6*self.E*self.I/(self.L**2)*self.lambx) , ((self.A * self.E/self.L -12*self.E*self.I/(self.L**3))*self.lambx*self.lamby ) , (self.A * self.E/self.L*self.lamby**2 + 12*self.E*self.I/(self.L**3)*self.lambx**2) , -(6*self.E*self.I/(self.L**2)*self.lambx)  ],
            [-(6*self.E*self.I/(self.L**2)*self.lamby) , (6*self.E*self.I/(self.L**2)*self.lambx) , (2*self.E*self.I/self.L) , (6*self.E*self.I/(self.L**2)*self.lamby) , -((6*self.E*self.I/(self.L**2)*self.lambx)) , (4*self.E*self.I/self.L) ]
            ])
    
    
        self.indeks = np.array([
            [(self.dNx-1 ,  self.dNx-1 ,  0 ,  0) ,  (self.dNx-1 ,  self.dNy-1 ,  0 ,  1) ,  (self.dNx-1 ,  self.dNz-1 ,  0 ,  2) , (self.dNx-1 ,  self.dFx-1 ,  0 ,  3) ,  (self.dNx-1 ,  self.dFy-1 ,  0 ,  4) ,  (self.dNx-1 ,  self.dFz-1 ,  0 ,  5)] , 
            [(self.dNy-1 ,  self.dNx-1 ,  1 ,  0) ,  (self.dNy-1 ,  self.dNy-1 ,  1 ,  1) ,  (self.dNy-1 ,  self.dNz-1 ,  1 ,  2) , (self.dNy-1 ,  self.dFx-1 ,  1 ,  3) ,  (self.dNy-1 ,  self.dFy-1 ,  1 ,  4) ,  (self.dNy-1 ,  self.dFz-1 ,  1 ,  5)] , 
            [(self.dNz-1 ,  self.dNx-1 ,  2 ,  0) ,  (self.dNz-1 ,  self.dNy-1 ,  2 ,  1) ,  (self.dNz-1 ,  self.dNz-1 ,  2 ,  2) , (self.dNz-1 ,  self.dFx-1 ,  2 ,  3) ,  (self.dNz-1 ,  self.dFy-1 ,  2 ,  4) ,  (self.dNz-1 ,  self.dFz-1 ,  2 ,  5)] , 
            [(self.dFx-1 ,  self.dNx-1 ,  3 ,  0) ,  (self.dFx-1 ,  self.dNy-1 ,  3 ,  1) ,  (self.dFx-1 ,  self.dNz-1 ,  3 ,  2) , (self.dFx-1 ,  self.dFx-1 ,  3 ,  3) ,  (self.dFx-1 ,  self.dFy-1 ,  3 ,  4) ,  (self.dFx-1 ,  self.dFz-1 ,  3 ,  5)] , 
            [(self.dFy-1 ,  self.dNx-1 ,  4 ,  0) ,  (self.dFy-1 ,  self.dNy-1 ,  4 ,  1) ,  (self.dFy-1 ,  self.dNz-1 ,  4 ,  2) , (self.dFy-1 ,  self.dFx-1 ,  4 ,  3) ,  (self.dFy-1 ,  self.dFy-1 ,  4 ,  4) ,  (self.dFy-1 ,  self.dFz-1 ,  4 ,  5)] , 
            [(self.dFz-1 ,  self.dNx-1 ,  5 ,  0) ,  (self.dFz-1 ,  self.dNy-1 ,  5 ,  1) ,  (self.dFz-1 ,  self.dNz-1 ,  5 ,  2) , (self.dFz-1 ,  self.dFx-1 ,  5 ,  3) ,  (self.dFz-1 ,  self.dFy-1 ,  5 ,  4) ,  (self.dFz-1 ,  self.dFz-1 ,  5 ,  5)]])

        self.D_indeks = np.array([
            [(self.dNx-1 ,  0)] , 
            [(self.dNy-1 ,  1)] , 
            [(self.dNz-1 ,  2)] , 
            [(self.dFx-1 ,  3)] , 
            [(self.dFy-1 ,  4)] , 
            [(self.dFz-1 ,  5)]
        ])

    def qatayici(self ,  D_global, Q_global):
        for i in self.D_indeks:
            for j ,  p in i:
                self.D[p] += D_global[j]
                self.Q[p] += Q_global[j]
        self.q += self.k_prime.dot(self.T).dot(self.D)

def uzay_cubuk_olusturucu(numara ,yakin ,  uzak ,  E,  A):
    uzay_cubuklar.append(uzay_cubuk_eleman(numara , yakin.x ,  yakin.y , yakin.z ,   uzak.x ,  uzak.y , uzak.z ,  yakin.d1x ,  yakin.d2y ,  yakin.d3z ,  uzak.d1x ,  uzak.d2y ,  uzak.d3z ,  E ,  A))
    uzay_yakin.append(yakin.numara)
    uzay_uzak.append(uzak.numara)

def cerceve_cubuk_olusturucu(numara , yakin ,  uzak ,  E ,  I ,  A):
    cerceve_cubuklar.append(cerceve_cubuk_eleman(numara , yakin.x ,  yakin.y ,  uzak.x ,  uzak.y ,  yakin.d1x ,  yakin.d2y ,  yakin.d3z ,  uzak.d1x ,  uzak.d2y ,  uzak.d3z ,  E ,  I ,  A))
    cerceve_yakin.append(yakin.numara)
    cerceve_uzak.append(uzak.numara)

def kafes_cubuk_olusturucu(numara , yakin , uzak , E , A):
    kafes_cubuklar.append(kafes_cubuk_eleman(numara , yakin.x , yakin.y , uzak.x , uzak.y , yakin.d1x , yakin.d2y , uzak.d1x , uzak.d2y , E , A))
    kafes_yakin.append(yakin.numara)
    kafes_uzak.append(uzak.numara)

def uzay_cubuk_eksenel_raporu(uzay_cubuk_durum_raporu , uzay_cubuk_q , uzay_cubuklar):
    for i in uzay_cubuklar:
        if (i.qF < 0) :
            a = "BASINÇ ELEMANI"
            uzay_cubuk_durum_raporu.append(a) 
            uzay_cubuk_q.append(i.qF[0]) 

        elif (i.qF >0):
            a = "ÇEKME ELEMANI"
            uzay_cubuk_durum_raporu.append(a)    
            uzay_cubuk_q.append(i.qF[0]) 

        else:
            a = "YuK ALMAMIŞ"
            uzay_cubuk_durum_raporu.append(a)
            uzay_cubuk_q.append(i.qF[0]) 

def kafes_cubuk_eksenel_raporu(kafes_cubuk_durum_raporu , kafes_cubuk_q , kafes_cubuklar):
    for i in kafes_cubuklar:
        if (i.qF < 0) :
            a = "BASINÇ ELEMANI"
            kafes_cubuk_durum_raporu.append(a) 
            kafes_cubuk_q.append(i.qF[0]) 

        elif (i.qF >0):
            a = "ÇEKME ELEMANI"
            kafes_cubuk_durum_raporu.append(a)    
            kafes_cubuk_q.append(i.qF[0]) 

        else:
            a = "YuK ALMAMIŞ"
            kafes_cubuk_durum_raporu.append(a)
            kafes_cubuk_q.append(i.qF[0]) 
                
           
def uzay_birinci_adim(Global_sistem ,  list):

    def K_arttir(K_global ,  k ,  indeks):
        for i in indeks:
            for j ,  p ,  r ,  t in i:
                K_global[j ,  p] += k[r ,  t]

    def Q_bilinen_ata(Q_global_bilinen ,  list):
        for j in list:
            Q_global_bilinen[j.dF_numarasi-1] += j.F
    Q_bilinen_ata(Global_sistem.Q_bilinen ,  yukler)
    

    def mesnet_cokmesi_ata(D_global_bilinen ,  list):
        for i in list:
            D_global_bilinen[i.dF_numarasi - (1 + dugum_sayisi*3 - tutulu_sayisi)] += i.cokme_miktari
    mesnet_cokmesi_ata(Global_sistem.D_bilinen ,  mesnet_cokme_listesi)

    for i in list:
        K_arttir(Global_sistem.K ,  i.k ,  i.indeks)


def cerceve_birinci_adim(Global_sistem ,  list):
    def K_arttir(K_global ,  k ,  indeks):
        for i in indeks:
            for j ,  p ,  r ,  t in i:
                K_global[j ,  p] += k[r ,  t]

    def Q_bilinen_ata(Q_global_bilinen ,  list):
        for j in list:
            Q_global_bilinen[j.dF_numarasi-1] += j.F
    Q_bilinen_ata(Global_sistem.Q_bilinen ,  yukler)
    
    def mesnet_cokmesi_ata(D_global_bilinen ,  list):
        for i in list:
            D_global_bilinen[i.dF_numarasi - (1 + dugum_sayisi*3 - tutulu_sayisi)] += i.cokme_miktari
    mesnet_cokmesi_ata(Global_sistem.D_bilinen ,  mesnet_cokme_listesi)

    for i in list:
        K_arttir(Global_sistem.K ,  i.k ,  i.indeks)
        
def kafes_birinci_adim(Global_sistem , list):

    def K_arttir(K_global , k , indeks):
        for i in indeks:
            for j , p , r , t in i:
                K_global[j , p] += k[r , t]

    def Q_bilinen_ata(Q_global_bilinen , list):
        for j in list:
            Q_global_bilinen[j.dF_numarasi-1] += j.F       
    Q_bilinen_ata(Global_sistem.Q_bilinen , yukler)
    
    def mesnet_cokmesi_ata(D_global_bilinen , list):
        for i in list:
            D_global_bilinen[i.dF_numarasi-(1 + dugum_sayisi*2 - tutulu_sayisi)] += i.cokme_miktari
    mesnet_cokmesi_ata(Global_sistem.D_bilinen , mesnet_cokme_listesi)
 
    for i in list:
       K_arttir(Global_sistem.K ,  i.k ,  i.indeks)
       

def uzay_Q_K_D_olusturucu(Global_sistem ,  list):
    def K_bolucu(K ,  s ,  n):
        def birinci():
            K11 = np.zeros((s ,  s))
            i = 0
            while i < s:
                K11[i] += K[i][0:s]
                i += 1
            return K11

        def ikinci():
            K12 = np.zeros((s ,  n-s))
            i = 0
            while i < s:
                K12[i] += K[i][s:n]
                i += 1
            return K12

        def ucuncu():
            K21 = np.zeros((n-s ,  s))
            j = 0
            i = s
            while i < n:
                K21[j] += K[i][0:s]
                i += 1
                j += 1
            return K21

        def dorduncu():
            K22 = np.zeros((n-s ,  n-s))
            j = 0
            i = s
            while i < n:
                K22[j] += K[i][s:n]
                i += 1
                j += 1
            return K22
        return birinci() ,  ikinci() ,  ucuncu() ,  dorduncu()

    Global_sistem.K11 ,  Global_sistem.K12 ,  Global_sistem.K21 ,  Global_sistem.K22 = K_bolucu(
        Global_sistem.K ,  Global_sistem.tutulu_olmayan ,  Global_sistem.n)
    K12_Dbilinen = np.dot(Global_sistem.K12 ,  Global_sistem.D_bilinen)
    K22_Dbilinen = np.dot(Global_sistem.K22 ,  Global_sistem.D_bilinen)
    Global_sistem.D_bilinmeyen = np.dot(np.linalg.pinv(Global_sistem.K11) ,  (Global_sistem.Q_bilinen - K12_Dbilinen))
    Global_sistem.Q_bilinmeyen = np.dot(Global_sistem.K21 ,  Global_sistem.D_bilinmeyen) + K22_Dbilinen

    Global_sistem.Q = np.vstack((Global_sistem.Q_bilinen ,  Global_sistem.Q_bilinmeyen))
    Global_sistem.D = np.vstack((Global_sistem.D_bilinmeyen ,  Global_sistem.D_bilinen))

    for m in uzay_cubuklar:
        m.qatayici(Global_sistem.D , Global_sistem.Q)

def cerceve_Q_K_D_olusturucu(Global_sistem ,  list):
    def K_bolucu(K ,  s ,  n):
        def birinci():
            K11 = np.zeros((s ,  s))
            i = 0
            while i < s:
                K11[i] += K[i][0:s]
                i += 1
            return K11

        def ikinci():
            K12 = np.zeros((s ,  n-s))
            i = 0
            while i < s:
                K12[i] += K[i][s:n]
                i += 1
            return K12

        def ucuncu():
            K21 = np.zeros((n-s ,  s))
            j = 0
            i = s
            while i < n:
                K21[j] += K[i][0:s]
                i += 1
                j += 1
            return K21

        def dorduncu():
            K22 = np.zeros((n-s ,  n-s))
            j = 0
            i = s
            while i < n:
                K22[j] += K[i][s:n]
                i += 1
                j += 1
            return K22
        return birinci() ,  ikinci() ,  ucuncu() ,  dorduncu()

    Global_sistem.K11 ,  Global_sistem.K12 ,  Global_sistem.K21 ,  Global_sistem.K22 = K_bolucu(
        Global_sistem.K ,  Global_sistem.tutulu_olmayan ,  Global_sistem.n)
    K12_Dbilinen = np.dot(Global_sistem.K12 ,  Global_sistem.D_bilinen)
    K22_Dbilinen = np.dot(Global_sistem.K22 ,  Global_sistem.D_bilinen)
    Global_sistem.D_bilinmeyen = np.dot(np.linalg.pinv(
        Global_sistem.K11) ,  (Global_sistem.Q_bilinen - K12_Dbilinen))
    Global_sistem.Q_bilinmeyen = np.dot(
        Global_sistem.K21 ,  Global_sistem.D_bilinmeyen) + K22_Dbilinen

    Global_sistem.Q = np.vstack((Global_sistem.Q_bilinen ,  Global_sistem.Q_bilinmeyen))
    Global_sistem.D = np.vstack((Global_sistem.D_bilinmeyen ,  Global_sistem.D_bilinen))

    for m in cerceve_cubuklar:
        m.qatayici(Global_sistem.D , Global_sistem.Q)
 
def kafes_Q_K_D_olusturucu(Global_sistem , list):
    def K_bolucu(K , s , n):
        def birinci():
            K11 = np.zeros((s , s))
            i = 0
            while i < s:
                K11[i] += K[i][0:s]  
                i += 1
            return K11
        def ikinci():
            K12 = np.zeros((s , n-s))
            i = 0
            while i < s:
                K12[i] += K[i][s:n]  
                i += 1
            return K12
        def ucuncu():
            K21 = np.zeros((n-s , s))
            j = 0
            i = s
            while i < n:
                K21[j] += K[i][0:s]  
                i += 1
                j += 1
            return K21
            
        def dorduncu():
            K22 = np.zeros((n-s , n-s))
            j = 0
            i = s
            while i < n:
                K22[j] += K[i][s:n]  
                i += 1      
                j += 1
            return K22
        return birinci() , ikinci() , ucuncu() , dorduncu()

    Global_sistem.K11  ,  Global_sistem.K12  ,  Global_sistem.K21  ,  Global_sistem.K22 = K_bolucu(Global_sistem.K , Global_sistem.tutulu_olmayan , Global_sistem.n)
    K12_Dbilinen = np.dot(Global_sistem.K12  ,  Global_sistem.D_bilinen)
    K22_Dbilinen = np.dot(Global_sistem.K22  ,  Global_sistem.D_bilinen)
    Global_sistem.D_bilinmeyen = np.dot(np.linalg.pinv(Global_sistem.K11) , (Global_sistem.Q_bilinen - K12_Dbilinen))
    Global_sistem.Q_bilinmeyen = np.dot(Global_sistem.K21 , Global_sistem.D_bilinmeyen) + K22_Dbilinen

    Global_sistem.Q = np.vstack((Global_sistem.Q_bilinen , Global_sistem.Q_bilinmeyen))
    Global_sistem.D = np.vstack((Global_sistem.D_bilinmeyen , Global_sistem.D_bilinen))

    for m in kafes_cubuklar:
        m.qatayici(Global_sistem.D , Global_sistem.Q)



def uzay_grafik(Global_sistem ,  list  ,  oran):
    def xy_bos():
        x = []
        y = []
        z = []
        return x ,  y , z
    def uzay_xy_stack(uzay_cubuk ,  x ,  y , z):
        x.append([uzay_cubuk.xN , uzay_cubuk.xF])
        y.append([uzay_cubuk.yN , uzay_cubuk.yF])
        z.append([uzay_cubuk.zN , uzay_cubuk.zF])
    x ,  y , z = xy_bos()
    for i in list:
        uzay_xy_stack(i , x , y , z)

        if i.qF < 0 :
            uzay_colors.append("o-r")
        elif i.qF ==0:
            uzay_colors.append("o-k")
        else:
            uzay_colors.append("o-g")
    y_deplase = []
    x_deplase = []
    z_deplase = []
    i = 0
    while i < Global_sistem.eleman_sayisi :
        for j in list :
            y_deplase.append([ float(y[i][0] + (j.D[1])*(oran)) , float(y[i][1] + (j.D[4])*(oran)) ])
            x_deplase.append([ float(x[i][0] + (j.D[0])*(oran)) , float(x[i][1] + (j.D[3])*(oran)) ])
            z_deplase.append([ float(z[i][0] + (j.D[2])*(oran)) , float(z[i][1] + (j.D[5])*(oran)) ])
            i += 1

    fig = plt.figure()
    ax = plt.axes(projection = "3d")
    fig.suptitle("SİSTEM ANALİZ SONUCU" , fontname="Times New Roman" ,  fontsize=20 ,  color="r")
    for k in range(len(x)):
        ax.plot3D(x[k] , y[k] , z[k] , "o-k" ,  linewidth=2 ,  markersize=15 ,  mfc="white")
        ax.plot3D(x_deplase[k] , y_deplase[k] , z_deplase[k] ,  uzay_colors[k] , linewidth=2 , markersize=15 ,  mec="r" ,  mfc="white")

    ax.grid(False) 
    ax.axis('off')
    ax.legend()
    plt.show()
    #plt.savefig(r'C:\Users\Msi\Desktop\Bitirme_Tezi\SONUÇLAR\Deformed Shape.png')
    plt.savefig('Deformed Shape.png')


def kafes_grafik (Global_sistem , list , oran):
    def xy_bos():
        x = []
        y = []
        z = []
        return x , y ,z
    def kafes_xy_stack(kafes_cubuk , x , y , z):
        x.append([kafes_cubuk.xN , kafes_cubuk.xF])
        y.append([kafes_cubuk.yN , kafes_cubuk.yF])
        z.append([0,0])
    x , y , z = xy_bos()

    x_dugum = []
    y_dugum = []
    def kafes_dugum_xy_stack(kafes_dugum, x,y):
        x.append(kafes_dugum.x-0.01)
        y.append(kafes_dugum.y-0.01)
        

    for i in kafes_dugum_listesi:
        kafes_dugum_xy_stack(i, x_dugum, y_dugum)


    for i in list:
        kafes_xy_stack(i , x , y ,z)

        if i.qF < 0 :
            kafes_colors.append("o-r")
        elif i.qF ==0:
            kafes_colors.append("o-k")
        else:
            kafes_colors.append("o-g")

    y_deplase = []
    x_deplase = []
    z_deplase = []
    i = 0
    while i < Global_sistem.eleman_sayisi :
        for j in list :
            y_deplase.append( [float(y[i][0] + (j.D[1])*(oran)) , float(y[i][1] + (j.D[3])*(oran)) ])
            x_deplase.append([ float(x[i][0] + (j.D[0])*(oran)) , float(x[i][1] + (j.D[2])*(oran)) ])
            z_deplase.append([0,0])
            i += 1

    fig1 , ax1 = plt.subplots(figsize = (15 , 10))
    for k in range(len(x)):
        ax1.plot(x[k] , y[k] , "grey" ,  linewidth= 4  ,  markersize = 10  ,  mfc="white")
        ax1.plot(x_deplase[k] , y_deplase[k] , kafes_colors[k] , linewidth= 4  ,  markersize = 10 ,  mec="r"  ,  mfc="white")

    for k in range(len(x_dugum)):
        ax1.text(x_dugum[k] , y_dugum[k] , str(kafes_dugum_numara_listesi[k]), fontweight = 'bold' , fontsize = 10)


    ax1.grid(False) 
    ax1.axis('off')
    ax1.legend()
    plt.show()


    fig = plt.figure()
    ax = plt.axes(projection = "3d")
    for k in range(len(x)):
        ax.plot3D(x[k] ,  z[k] , y[k] , "o-k" ,  linewidth=2 ,  markersize=15 ,  mfc="white")
        ax.plot3D(x_deplase[k] , z_deplase[k] , y_deplase[k],  kafes_colors[k] , linewidth=2 , markersize=15 ,  mec="r" ,  mfc="white")
    ax.grid(False) 
    ax.axis('off')
    ax.legend()

    plt.show()
    fig1.savefig('Deformed Shape.png')
    fig.savefig('Deformed Shape.pdf')


def cerceve_grafik (Global_sistem , list , oran):
    def xy_bos():
        x = []
        y = []
        return x , y

    x_dugum = []
    y_dugum = []

    def cerceve_xy_stack(cerceve_cubuk , x , y):
        x.append([cerceve_cubuk.xN , cerceve_cubuk.xF])
        y.append([cerceve_cubuk.yN , cerceve_cubuk.yF])
    x , y = xy_bos()

    def cerceve_dugum_xy_stack(cerceve_dugum, x,y):
        x.append(cerceve_dugum.x-0.11)
        y.append(cerceve_dugum.y-0.11)


    for i in cerceve_dugum_listesi:
        cerceve_dugum_xy_stack(i, x_dugum, y_dugum)


    for i in list:
        cerceve_xy_stack(i , x , y)

    y_deplase = []
    x_deplase = []
    z_deplase = []

    i = 0
    while i < Global_sistem.eleman_sayisi :
        for j in list :
            y_deplase.append([ float(y[i][0] + (j.D[1])*(oran)) , float(y[i][1] + (j.D[4])*(oran)) ])
            x_deplase.append([ float(x[i][0] + (j.D[0])*(oran)) , float(x[i][1] + (j.D[3])*(oran)) ])
            z_deplase.append([0,0])
            i += 1

    fig = plt.figure()
    ax = plt.axes(projection = "3d")
    fig.suptitle("SİSTEM ANALİZ SONUCU")
    for k in range(len(x)):
        ax.plot3D(x[k] ,  z_deplase[k] , y[k] , "o-k" ,  linewidth=2 ,  markersize=15 ,  mfc="white")
        ax.plot3D(x_deplase[k] , z_deplase[k] , y_deplase[k],  "o-r" , linewidth=2 , markersize=15 ,  mec="r" ,  mfc="white")
    ax.legend()

    fig1 , ax1 = plt.subplots(figsize = (15 , 10))
    fig1.suptitle("SİSTEM ANALİZ SONUÇLARI")

    #ax.grid(False) 
    #ax.axis('off')


    for k in range(len(x)):
        ax1.plot(x[k] , y[k] , "o-k" ,  linewidth= 4  ,  markersize = 15  ,  mfc="white")
        ax1.plot(x_deplase[k] , y_deplase[k] , "o--r" , linewidth= 3  ,  markersize = 15 ,  mec="r"  ,  mfc="white")
    for k in range(len(x_dugum)):
        ax1.text(x_dugum[k] , y_dugum[k] , str(cerceve_dugum_numara_listesi[k]), fontweight = 'bold' , fontsize = 10)

    ax1.legend()
    ax1.grid(False) 
    ax1.axis('off')
    plt.show()
    fig1.savefig('Deformed Shape.png')
    
def yuk_olusturucu(dF_numarasi ,  yuk_KN):
    yukler.append(yuk(dF_numarasi ,  yuk_KN))


def mesnet_cokmesi_olusturucu(dF_numarasi ,  cokme_miktari):
    mesnet_cokme_listesi.append(mesnet_cokmesi(dF_numarasi ,  cokme_miktari))


def kafes_joimt_output(Global_sistem , dugum_list , dugum_numara_list ):
    dugum_numara_list.sort()
    for i in dugum_list:

        Global_sistem.Fx[i.numara-1] += Global_sistem.Q[i.d1x-1]
        Global_sistem.Fy[i.numara-1] += Global_sistem.Q[i.d2y-1]

        Global_sistem.Ux[i.numara-1] += Global_sistem.D[i.d1x-1]
        Global_sistem.Uy[i.numara-1] += Global_sistem.D[i.d2y-1]   
   
    data = {
        "JOİNT" : dugum_numara_list,
        "Fx (KN)"    :Global_sistem.Fx,
        "Fy (KN)"    : Global_sistem.Fy,
        "Fz (KN)"    : Global_sistem.Fz,
        "Ux (m)"     : Global_sistem.Ux,
        "Uy (m)"     : Global_sistem.Uy,
        "Uz (m)"     : Global_sistem.Uz }
    df = pd.DataFrame(data)
    #return df.to_excel(r'C:\Users\Msi\Desktop\Bitirme_Tezi\SONUÇLAR\Joint Output.xlsx')
    return df.to_excel('Joint Output.xlsx')

def kafes_element_output(cubuk_list , Global_sistem):

    for i in cubuk_list:
        kafes_FNx.append(i.Q[0][0])
        kafes_FNy.append(i.Q[1][0])
        kafes_FFx.append(i.Q[2][0])
        kafes_FFy.append(i.Q[3][0])

        Global_sistem.UNx[i.numara-1] += i.D[0]
        Global_sistem.UNy[i.numara-1] += i.D[1]
        Global_sistem.UFx[i.numara-1] += i.D[2]
        Global_sistem.UFy[i.numara-1] += i.D[3]
      
    data = {
        "FRAME" : kafes_cubuk_numara_listesi,
        "YAKİN DN" : kafes_yakin,
        "UZAK DN" : kafes_uzak,
        "qF (KN)"      : kafes_cubuk_q,
        "FNx (KN)"      : kafes_FNx,
        "FNy (KN)"      : kafes_FNy,        
        "FFx (KN)"      : kafes_FFx,           
        "FFy (KN)"      : kafes_FFy,
        "UNx (m)"      : Global_sistem.UNx, 
        "UNy (m)"      : Global_sistem.UNy,             
        "UFx (m)"      : Global_sistem.UFx, 
        "UFy (m)"      : Global_sistem.UFy,     
        "DURUM"        : kafes_cubuk_durum_raporu}
        
    df = pd.DataFrame(data)     
    #return df.to_excel(r'C:\Users\Msi\Desktop\Bitirme_Tezi\SONUÇLAR\Element Output.xlsx')
    return df.to_excel('Joint Output.xlsx')   

def uzay_joimt_output(Global_sistem , dugum_list , dugum_numara_list ):
    dugum_numara_list.sort()
    for i in dugum_list:
        Global_sistem.Fx[i.numara-1] += Global_sistem.Q[i.d1x-1]
        Global_sistem.Fy[i.numara-1] += Global_sistem.Q[i.d2y-1]
        Global_sistem.Fz[i.numara-1] += Global_sistem.Q[i.d3z-1]

        Global_sistem.Ux[i.numara-1] += Global_sistem.D[i.d1x-1]
        Global_sistem.Uy[i.numara-1] += Global_sistem.D[i.d2y-1]   
        Global_sistem.Uz[i.numara-1] += Global_sistem.D[i.d3z-1]
    data = {
        "JOİNT" : dugum_numara_list,
        "Fx (KN)"    : Global_sistem.Fx,
        "Fy (KN)"    : Global_sistem.Fy,
        "Fz (KN)"    : Global_sistem.Fz,
        "Ux (m)"     : Global_sistem.Ux,
        "Uy (m)"     : Global_sistem.Uy,
        "Uz (m)"     : Global_sistem.Uz }
    df = pd.DataFrame(data)
    #return df.to_excel(r'C:\Users\Msi\Desktop\Bitirme_Tezi\SONUÇLAR\Joint Output.xlsx')
    return df.to_excel('Joint Output.xlsx')

def uzay_element_output(cubuk_list , Global_sistem):
    for i in cubuk_list:
        uzay_FNx.append(i.Q[0][0])
        uzay_FNy.append(i.Q[1][0])
        uzay_FNz.append(i.Q[2][0])
        uzay_FFx.append(i.Q[3][0])
        uzay_FFy.append(i.Q[4][0])
        uzay_FFz.append(i.Q[5][0])
        
        Global_sistem.UNx[i.numara-1] += i.D[0]
        Global_sistem.UNy[i.numara-1] += i.D[1]
        Global_sistem.UNz[i.numara-1] += i.D[2]
        Global_sistem.UFx[i.numara-1] += i.D[3]
        Global_sistem.UFy[i.numara-1] += i.D[4]
        Global_sistem.UFz[i.numara-1] += i.D[5]       

    data = {
        "FRAME" : uzay_cubuk_numara_listesi,
        "YAKİN DN" : uzay_yakin,
        "UZAK DN" : uzay_uzak,
        "qF (KN)"      : uzay_cubuk_q,
        "FNx (KN)"      : uzay_FNx,
        "FNy (KN)"      : uzay_FNy,
        "FNz (KN)"      : uzay_FNz,          
        "FFx (KN)"      : uzay_FFx,           
        "FFy (KN)"      : uzay_FFy,
        "FFz (KN)"      : uzay_FFz,
        "UNx (m)"      : Global_sistem.UNx, 
        "UNy (m)"      : Global_sistem.UNy,   
        "UNz (m)"      : Global_sistem.UNz,               
        "UFx (m)"      : Global_sistem.UFx, 
        "UFy (m)"      : Global_sistem.UFy,  
        "UFz (m)"      : Global_sistem.UFz,             
        "DURUM"        : uzay_cubuk_durum_raporu
            }
    df = pd.DataFrame(data)       
    #return df.to_excel(r'C:\Users\Msi\Desktop\Bitirme_Tezi\SONUÇLAR\Element Output.xlsx')
    return df.to_excel('Element Output.xlsx')

def cerceve_element_output(cubuk_list , Global_sistem):
  
    for i in cubuk_list:
        cerceve_FNx.append(i.q[0][0])
        cerceve_FNy.append(i.q[1][0])
        cerceve_FNz.append(i.q[2][0])
        cerceve_FFx.append(i.q[3][0])
        cerceve_FFy.append(i.q[4][0])
        cerceve_FFz.append(i.q[5][0])

        Global_sistem.UNx[i.numara-1] += i.D[0]
        Global_sistem.UNy[i.numara-1] += i.D[1]
        Global_sistem.UNz[i.numara-1] += i.D[2]
        Global_sistem.UFx[i.numara-1] += i.D[3]
        Global_sistem.UFy[i.numara-1] += i.D[4]
        Global_sistem.UFz[i.numara-1] += i.D[5]    
        
    data = {
        "FRAME" : cerceve_cubuk_numara_listesi,
        "YAKİN DN" : cerceve_yakin,
        "UZAK DN" : cerceve_uzak,
        "FNx (KN)"      : cerceve_FNx,
        "FNy (KN)"      : cerceve_FNy,
        "FNz (KNm)"      : cerceve_FNz,          
        "FFx (KN)"      : cerceve_FFx,           
        "FFy (KN)"      : cerceve_FFy,
        "FFz (KNm)"      : cerceve_FFz,
        "UNx (m)"      : Global_sistem.UNx, 
        "UNy (m)"      : Global_sistem.UNy,   
        "UNz (m)"      : Global_sistem.UNz,               
        "UFx (m)"      : Global_sistem.UFx, 
        "UFy (m)"      : Global_sistem.UFy,  
        "UFz (rad)"      : Global_sistem.UFz,             
            }
    df = pd.DataFrame(data)       
    #return df.to_excel(r'C:\Users\Msi\Desktop\Bitirme_Tezi\SONUÇLAR\Element Output.xlsx')
    return df.to_excel('Element Output.xlsx')


def cerceve_joimt_output(Global_sistem , dugum_list , dugum_numara_list ):
    dugum_numara_list.sort()
    for i in dugum_list:
        Global_sistem.Fx[i.numara-1] += Global_sistem.Q[i.d1x-1]
        Global_sistem.Fy[i.numara-1] += Global_sistem.Q[i.d2y-1]
        Global_sistem.Fz[i.numara-1] += Global_sistem.Q[i.d3z-1]

        Global_sistem.Ux[i.numara-1] += Global_sistem.D[i.d1x-1]
        Global_sistem.Uy[i.numara-1] += Global_sistem.D[i.d2y-1]   
        Global_sistem.Uz[i.numara-1] += Global_sistem.D[i.d3z-1]
    data = {
        "JOİNT" : dugum_numara_list,
        "Fx (KN)"    :Global_sistem.Fx,
        "Fy (KN)"    : Global_sistem.Fy,
        "Fz (KNm)"    : Global_sistem.Fz,
        "Ux (m)"     : Global_sistem.Ux,
        "Uy (m)"     : Global_sistem.Uy,
        "Uz (rad)"     : Global_sistem.Uz }
    df = pd.DataFrame(data)
    #return df.to_excel(r'C:\Users\Msi\Desktop\Bitirme_Tezi\SONUÇLAR\Joint Output.xlsx')
    return df.to_excel('Joint Output.xlsx')



def kafes_Base_Reaktions(Global_sistem , dugum_list , dugum_numara_list , mesnet_list):

    for i  in mesnet_list :

        dugum_numara_list.append(i.numara)
        j = dugum_numara_list.index(i.numara)
        Global_sistem.Mesnet_Fx[j] += Global_sistem.Q[i.d1x - 1] 
        Global_sistem.Mesnet_Fy[j] += Global_sistem.Q[i.d2y - 1] 
    
        Global_sistem.Mesnet_Ux[j] += Global_sistem.D[i.d1x - 1] 
        Global_sistem.Mesnet_Uy[j] += Global_sistem.D[i.d2y - 1] 

    data = {
        "BASE JOİNT" : dugum_numara_list,
        "Fx (KN)"    :Global_sistem.Mesnet_Fx,
        "Fy (KN)"    : Global_sistem.Mesnet_Fy,
        "Fz (KN)"    : Global_sistem.Mesnet_Fz,
        "Ux (m)"     : Global_sistem.Mesnet_Ux,
        "Uy (m)"     : Global_sistem.Mesnet_Uy,
        "Uz (rad)"     : Global_sistem.Mesnet_Uz }
    df = pd.DataFrame(data)
    #return df.to_excel(r'C:\Users\Msi\Desktop\Bitirme_Tezi\SONUÇLAR\Base Reactions.xlsx')
    return df.to_excel('Base Reactions.xlsx')

def uzay_Base_Reaktions(Global_sistem , dugum_list , dugum_numara_list , mesnet_list):

    for i  in mesnet_list :

        dugum_numara_list.append(i.numara)
        j = dugum_numara_list.index(i.numara)
        Global_sistem.Mesnet_Fx[j] += Global_sistem.Q[i.d1x - 1] 
        Global_sistem.Mesnet_Fy[j] += Global_sistem.Q[i.d2y - 1] 
        Global_sistem.Mesnet_Fz[j] += Global_sistem.Q[i.d3z - 1]             
    
        Global_sistem.Mesnet_Ux[j] += Global_sistem.D[i.d1x - 1] 
        Global_sistem.Mesnet_Uy[j] += Global_sistem.D[i.d2y - 1] 
        Global_sistem.Mesnet_Uz[j] += Global_sistem.D[i.d3z - 1]         
    data = {
        "BASE JOİNT" : dugum_numara_list,
        "Fx (KN)"    :Global_sistem.Mesnet_Fx,
        "Fy (KN)"    : Global_sistem.Mesnet_Fy,
        "Fz (KN)"    : Global_sistem.Mesnet_Fz,
        "Ux (m)"     : Global_sistem.Mesnet_Ux,
        "Uy (m)"     : Global_sistem.Mesnet_Uy,
        "Uz (m)"     : Global_sistem.Mesnet_Uz }
    df = pd.DataFrame(data)
    #return df.to_excel(r'C:\Users\Msi\Desktop\Bitirme_Tezi\SONUÇLAR\Base Reactions.xlsx')
    return df.to_excel('Base Reactions.xlsx')


def cerceve_Base_Reaktions(Global_sistem , dugum_list , dugum_numara_list , mesnet_list):

    for i  in mesnet_list :

        dugum_numara_list.append(i.numara)
        j = dugum_numara_list.index(i.numara)
        Global_sistem.Mesnet_Fx[j] += Global_sistem.Q[i.d1x - 1] 
        Global_sistem.Mesnet_Fy[j] += Global_sistem.Q[i.d2y - 1] 
        Global_sistem.Mesnet_Fz[j] += Global_sistem.Q[i.d3z - 1]             
    
        Global_sistem.Mesnet_Ux[j] += Global_sistem.D[i.d1x - 1] 
        Global_sistem.Mesnet_Uy[j] += Global_sistem.D[i.d2y - 1] 
        Global_sistem.Mesnet_Uz[j] += Global_sistem.D[i.d3z - 1]         
    data = {
        "BASE JOİNT" : dugum_numara_list,
        "Fx (KN)"    :Global_sistem.Mesnet_Fx,
        "Fy (KN)"    : Global_sistem.Mesnet_Fy,
        "Fz (KNm)"    : Global_sistem.Mesnet_Fz,
        "Ux (m)"     : Global_sistem.Mesnet_Ux,
        "Uy (m)"     : Global_sistem.Mesnet_Uy,
        "Uz (rad)"     : Global_sistem.Mesnet_Uz }
    df = pd.DataFrame(data)
    #return df.to_excel(r'C:\Users\Msi\Desktop\Bitirme_Tezi\SONUÇLAR\Base Reactions.xlsx')
    return df.to_excel('Base Reactions.xlsx')






def uzay_calistirici(sistem ,  t0 ,  ORAN):
    uzay_birinci_adim(sistem ,  uzay_cubuklar)
    uzay_Q_K_D_olusturucu(sistem ,  uzay_cubuklar)
    uzay_cubuk_eksenel_raporu(uzay_cubuk_durum_raporu  ,  uzay_cubuk_q  ,  uzay_cubuklar)    
    uzay_joimt_output(sistem , uzay_dugum_listesi , uzay_dugum_numara_listesi)
    uzay_element_output(uzay_cubuklar, sistem)
    uzay_Base_Reaktions(sistem , uzay_dugum_listesi , uzay_menset_numara_listesi , uzay_mesnetler)

    uzay_grafik(sistem ,  uzay_cubuklar ,   ORAN)

def cerceve_calistirici(sistem ,  t0 ,  ORAN):

    cerceve_birinci_adim(sistem ,  cerceve_cubuklar)
    cerceve_Q_K_D_olusturucu(sistem ,  cerceve_cubuklar)
    cerceve_joimt_output(sistem , cerceve_dugum_listesi , cerceve_dugum_numara_listesi)
    cerceve_Base_Reaktions(sistem , cerceve_dugum_listesi , cerceve_menset_numara_listesi , cerceve_mesnetler)
    cerceve_element_output(cerceve_cubuklar, sistem)

    cerceve_grafik(sistem ,  cerceve_cubuklar ,   ORAN)



def kafes_calistirici(sistem , t0 , ORAN):
    kafes_birinci_adim(sistem  ,  kafes_cubuklar)
    kafes_Q_K_D_olusturucu(sistem , kafes_cubuklar) 
    kafes_cubuk_eksenel_raporu(kafes_cubuk_durum_raporu  ,  kafes_cubuk_q  ,  kafes_cubuklar)
    kafes_joimt_output(sistem , kafes_dugum_listesi , kafes_dugum_numara_listesi)
    kafes_element_output(kafes_cubuklar, sistem)
    kafes_Base_Reaktions(sistem , kafes_dugum_listesi , kafes_menset_numara_listesi , kafes_mesnetler)

    kafes_grafik(sistem  ,  kafes_cubuklar  , ORAN)










def soru_sor():
    
    soru1 = int(input("Çözün yapılmasını istediğiniz sistemi seçiniz.(1,2 veya 3 ü tuşlayınız.) \n1) Kafes sistem \n2) Uzay Kaefs sistem \n3) Çerçeve sistem\n: "))
    if soru1 == 1:
        return "Kafes"
    elif soru1 == 2:
        return "Uzay Kafes"
    elif soru1 == 3:
        return  "Çerçeve"
    else :
        "Lütfen geçerli bir giriş yapınız."


def deplasman_buyutme_katsayisi():
    print("")
    soru3 = float(input("Deplasmanı kaç kat büyütmek istersiniz? (NOT: BU KATSAYI YALNIZCA DEFORME OLMUŞ ŞEKİL İÇİN GEÇERLİDİR.\n: "))
    print("")
    return soru3

def dugum_sayisi_sorusu():
    print("")
    soru4 = int(input("Mesnet Hariç Düğüm noktası sayısı kaçtır? \n: "))
    print("")

    return soru4

def eleman_sayisi_sorusu():
    print("")
    soru4 = int(input("Toplam eleman sayısı kaçtır? \n: "))
    print("")

    return soru4


def mesnet_sayisi_sorusu():
    print("")
    soru4 = int(input("Toplam mesnet sayısı kaçtır? \n: "))
    print("")

    return soru4



i=0
def dugum_data_sor(i , sistem_datası , mesnet_haric_dugum_sayisi ):

    if sistem_datası == "Kafes":
        j = 0
        while j < mesnet_haric_dugum_sayisi:
            print("")
            print("")
            print("Mesnet Hariç Düğüm noktası tanımlama kısmındasınız.")
            print("")

            print(f"Şu anda {j+1}. düğüm noktası için giriş yapıyorsunuz. ")
            a = float(input("Düğün Noktasının Numarasını Giriniz: \n: "))
            b = float(input("Düğüm noktasının x koordinatını giriniz (m) \n: "))
            c = float(input("Düğüm noktasının y koordinatını giriniz. (m) \n: "))
            kafes_dugum_noktasi(a , b , c , i+1 , i+2)
            i +=2
            j+=1
            print(f"Bu ana kadar {j+1} adet düğüm noktası için giriş yaptınız.")
            print("")
            print("")



            
    elif sistem_datası == "Uzay Kafes":
        j = 0
        while j < mesnet_haric_dugum_sayisi:
            print("")
            print("")
            print("Düğüm noktası tanımlama kısmındasınız.")
            print("")

            print(f"Şu anda {j+1}. düğüm noktası için giriş yapıyorsunuz. ")
            a = float(input("Düğün Noktasının Numarasını Giriniz: \n: "))
            b = float(input("Düğüm noktasının x koordinatını giriniz (m) \n: "))
            c = float(input("Düğüm noktasının y koordinatını giriniz. (m) \n: "))
            d = float(input("Düğüm noktasının z koordinatını giriniz. (m) \n: "))
            uzay_dugum_noktasi(a , b, c, d , i+1 , i+2 , i+3 )
            i +=3
            j+=1  
            print(f"Bu ana kadar {j+1} adet düğüm noktası için giriş yaptınız.")
            print("")
            print("")      

    elif sistem_datası == "Çerçeve":
        j = 0
        while j < mesnet_haric_dugum_sayisi:
            print("")
            print("")
            print("Düğüm noktası tanımlama kısmındasınız.")
            print("")
            print(f"Şu anda {j+1}. düğüm noktası için giriş yapıyorsunuz. ")
            a = float(input("Düğün Noktasının Numarasını Giriniz: \n: "))
            b = float(input("Düğüm noktasının x koordinatını giriniz (m) \n: "))
            c = float(input("Düğüm noktasının y koordinatını giriniz. (m) \n: "))
            cerceve_dugum_noktasi(a , b , c , i+1 , i+2, i+3)
            print(f"Bu ana kadar {j+1} adet düğüm noktası için giriş yaptınız.")
            print("")
            print("")

            i += 3
            j += 1        
            




def mesnet_data_sor(mesnet_haric_dugum_sayisi, sistem_datası , mesnet_sayisi, dugum_sayisi):

    print("")
    print("")
    print("Mesnet atama aşamasındasınız.")


    if sistem_datası == "Kafes":
        tutulu = 0
        i = mesnet_haric_dugum_sayisi*2
        j = 0
        serbestlik_sayisi = dugum_sayisi *2

        while j < mesnet_sayisi:
            print(" ")
            print(" ")
            print(j+1,". mesnet için")
            a = float(input("Düğün Noktasının Numarasını Giriniz: \n: "))
            b = float(input("Mesnet noktasının x koordinatını giriniz (m) \n: "))
            c = float(input("Mesnet noktasının y koordinatını giriniz. (m) \n: "))

            d = int(input("x eksenindeki serbestlik tutulu mu? \n Evet ise 1 i Hayır ise 0 ı tuşlayınız.\n: "))
            e = int(input("y eksenindeki serbestlik tutulu mu? \n Evet ise 1 i Hayır ise 0 ı tuşlayınız.\n: "))
            print(" ")
            print(" ")


            if d == 1 and e == 0 :
                d_indeks = serbestlik_sayisi
                serbestlik_sayisi -= 1
                e_indeks = i+1
                i += 1
                tutulu += 1
            elif d == 0 and e == 1:
                e_indeks = serbestlik_sayisi
                serbestlik_sayisi -= 1
                d_indeks = i+1
                i += 1
                tutulu += 1
                
            elif d == 1 and e == 1:
                e_indeks = serbestlik_sayisi
                serbestlik_sayisi -= 1
                d_indeks = serbestlik_sayisi
                serbestlik_sayisi -= 1
                tutulu += 2
                
            kafes_mesnet(a, b, c, d_indeks, e_indeks)
            j+=1
        return tutulu

    elif sistem_datası == "Çerçeve":
        tutulu = 0
        i = mesnet_haric_dugum_sayisi*3
        j = 0
        serbestlik_sayisi = dugum_sayisi *3

        while j < mesnet_sayisi:
            print(" ")
            print(" ")
            print(j+1,". mesnet için")
            a = float(input("Düğün Noktasının Numarasını Giriniz: \n: "))
            b = float(input("Mesnet noktasının x koordinatını giriniz (m) \n: "))
            c = float(input("Mesnet noktasının y koordinatını giriniz. (m) \n: "))


            d = int(input("x eksenindeki serbestlik tutulu mu? \n Evet ise 1 i Hayır ise 0 ı tuşlayınız.\n: "))
            e = int(input("y eksenindeki serbestlik tutulu mu? \n Evet ise 1 i Hayır ise 0 ı tuşlayınız.\n: "))
            f = int(input("z eksenindeki serbestlik tutulu mu? \n Evet ise 1 i Hayır ise 0 ı tuşlayınız.\n: "))
            print(" ")
            print(" ")

            if d == 1 and e == 0 and f == 0 :
                d_indeks = serbestlik_sayisi
                serbestlik_sayisi -= 1
                e_indeks = i+1
                f_indeks = i+2
                i += 2
                tutulu += 1
            elif d == 0 and e == 1 and f ==0:
                e_indeks = serbestlik_sayisi
                serbestlik_sayisi -= 1
                d_indeks = i+1
                f_indeks = i+2
                i += 2
                tutulu += 1
            
            elif d == 0 and e == 0 and f ==1:
                f_indeks = serbestlik_sayisi
                serbestlik_sayisi -=1
                d_indeks = i+1
                e_indeks = i+2
                i+=2
                tutulu += 1
                
                
            elif d==1 and e ==1 and f == 0:
                e_indeks = serbestlik_sayisi
                serbestlik_sayisi -=1
                d_indeks = serbestlik_sayisi 
                serbestlik_sayisi -=1
                f_indeks = i+1
                i+=1
                tutulu +=2
                
            elif d==1 and e==0 and f==1:
                f_indeks = serbestlik_sayisi
                serbestlik_sayisi -=1
                d_indeks = serbestlik_sayisi 
                serbestlik_sayisi -=1
                e_indeks = i+1
                i +=1
                tutulu +=2                
            
            elif d==0 and e==1 and f ==1:
                f_indeks = serbestlik_sayisi
                serbestlik_sayisi -=1
                e_indeks = serbestlik_sayisi 
                serbestlik_sayisi -=1
                d_indeks = i+1
                i +=1
                tutulu +=2                          
            
            elif d== 1 and e==1 and f==1:
                f_indeks = serbestlik_sayisi
                serbestlik_sayisi -=1
                e_indeks = serbestlik_sayisi 
                serbestlik_sayisi -=1
                d_indeks = serbestlik_sayisi
                serbestlik_sayisi -=1
                tutulu +=3                          
                
            cerceve_mesnet(a, b, c, d_indeks, e_indeks, f_indeks)

            j+=1
        return tutulu        

    elif sistem_datası == "Uzay Kafes":
        tutulu = 0
        i = mesnet_haric_dugum_sayisi*3
        j = 0
        serbestlik_sayisi = dugum_sayisi *3

        while j < mesnet_sayisi:
            print(" ")
            print(" ")
            print(j+1,". mesnet için")
            a = float(input("Düğün Noktasının Numarasını Giriniz: \n: "))
            b = int(input("Mesnet noktasının x koordinatını giriniz (m) \n: "))
            c = float(input("Mesnet noktasının y koordinatını giriniz. (m) \n: "))
            g = float(input("Mesnet noktasının z koordinatını giriniz. (m) \n: "))

            d = int(input("x eksenindeki serbestlik tutulu mu? \n Evet ise 1 i Hayır ise 0 ı tuşlayınız.\n: "))
            e = int(input("y eksenindeki serbestlik tutulu mu? \n Evet ise 1 i Hayır ise 0 ı tuşlayınız.\n: "))
            f = int(input("z eksenindeki serbestlik tutulu mu? \n Evet ise 1 i Hayır ise 0 ı tuşlayınız.\n: "))
            print(" ")
            print(" ")

            if d == 1 and e == 0 and f == 0 :
                d_indeks = serbestlik_sayisi
                serbestlik_sayisi -= 1
                e_indeks = i+1
                f_indeks = i+2
                i += 2
                tutulu += 1
            elif d == 0 and e == 1 and f ==0:
                e_indeks = serbestlik_sayisi
                serbestlik_sayisi -= 1
                d_indeks = i+1
                f_indeks = i+2
                i += 2
                tutulu += 1
            
            elif d == 0 and e == 0 and f ==1:
                f_indeks = serbestlik_sayisi
                serbestlik_sayisi -=1
                d_indeks = i+1
                e_indeks = i+2
                i+=2
                tutulu += 1
                
                
            elif d==1 and e ==1 and f == 0:
                e_indeks = serbestlik_sayisi
                serbestlik_sayisi -=1
                d_indeks = serbestlik_sayisi 
                serbestlik_sayisi -=1
                f_indeks = i+1
                i+=1
                tutulu +=2
                
            elif d==1 and e==0 and f==1:
                f_indeks = serbestlik_sayisi
                serbestlik_sayisi -=1
                d_indeks = serbestlik_sayisi 
                serbestlik_sayisi -=1
                e_indeks = i+1
                i +=1
                tutulu +=2                
            
            elif d==0 and e==1 and f ==1:
                f_indeks = serbestlik_sayisi
                serbestlik_sayisi -=1
                e_indeks = serbestlik_sayisi 
                serbestlik_sayisi -=1
                d_indeks = i+1
                i +=1
                tutulu +=2                          
            
            
            
            elif d== 1 and e==1 and f==1:
                f_indeks = serbestlik_sayisi
                serbestlik_sayisi -=1
                e_indeks = serbestlik_sayisi 
                serbestlik_sayisi -=1
                d_indeks = serbestlik_sayisi
                serbestlik_sayisi -=1
                
                tutulu +=3                          
                
                
            uzay_mesnet(a, b, c, g, d_indeks, e_indeks, f_indeks)
            j+=1
        return tutulu        





def cubuk_data_sor(cubuk_adedi, sistem_datası , dugum_listesi):
    j  = 0
    print(" ")
    print(" ")
    print("Çubuk eleman oluşturma aşamasındasınız.")
    print(" ")

    print(cubuk_adedi," adet çubuk eleman için giriş yapacaksınız.")
    c = int(input("E,I ve A tüm kesitlerde aynı mı? Evet için 1 , Hayır için 0 ı tuşlayınız. \n: "))    
    print(" ")
    print(" ")

    
    
    if sistem_datası == "Kafes":
        
        if c == 1:
            E = float(input("Elastisite modülünü giriniz. (MPa) \n: "))
            A = float(input("Kesit alanı giriniz. (m2) \n: "))
                    
            while j < cubuk_adedi:
                print(" ")
                print(" ")
                print(j+1,". çubuk eleman için")
                aa = int(input("Yakın düğüm noktasının numarasını giriniz. \n: "))
                bb = int(input("Uzak düğüm noktasının numarasını giriniz. \n: "))
                print(" ")
                print(" ")
    
                for i in dugum_listesi:
                    if aa == i.numara:
                        a_indeks = dugum_listesi.index(i)
                    if bb == i.numara:
                        b_indeks = dugum_listesi.index(i)
        
                kafes_cubuk_olusturucu(j+1, dugum_listesi[a_indeks], dugum_listesi[b_indeks], E, A)
                j +=1
                
                    
        elif c == 0:
            while j < cubuk_adedi:
                print(" ")
                print(" ")

                print(j+1,". çubuk eleman için")
                aa = int(input("Yakın düğüm noktasının numarasını giriniz. \n: "))
                bb = int(input("Uzak düğüm noktasının numarasını giriniz. \n: "))
                E = float(input("Elastisite modülünü giriniz. (MPa) \n: "))
                A = float(input("Kesit alanı giriniz. (m2) \n: "))
                print(" ")
                print(" ")

                    
                for i in dugum_listesi:
                    if aa == i.numara:
                        a_indeks = dugum_listesi.index(i)
                        if bb == i.numara:
                            b_indeks = dugum_listesi.index(i)
        
                kafes_cubuk_olusturucu(j+1, dugum_listesi[a_indeks], dugum_listesi[b_indeks], E, A)
                j +=1  

    elif sistem_datası == "Uzay Kafes":

            
        if c == 1:
            E = float(input("Elastisite modülünü giriniz. (MPa) \n: "))
            A = float(input("Kesit alanı giriniz. (m2) \n: "))
                    
            while j < cubuk_adedi:
                print(" ")
                print(" ")
                print(j+1,". çubuk eleman için")
                aa = int(input("Yakın düğüm noktasının numarasını giriniz. \n: "))
                bb = int(input("Uzak düğüm noktasının numarasını giriniz. \n: "))
                print(" ")
                print(" ")
    
                for i in dugum_listesi:
                    if aa == i.numara:
                        a_indeks = dugum_listesi.index(i)
                    if bb == i.numara:
                        b_indeks = dugum_listesi.index(i)
        
                uzay_cubuk_olusturucu(j+1, dugum_listesi[a_indeks], dugum_listesi[b_indeks], E, A)
                j +=1
                
                    
        elif c == 0:
            while j < cubuk_adedi:
                print(" ")
                print(" ")
                print(j+1,". çubuk eleman için")
                aa = int(input("Yakın düğüm noktasının numarasını giriniz. \n: "))
                bb = int(input("Uzak düğüm noktasının numarasını giriniz. \n: "))
                E = float(input("Elastisite modülünü giriniz. (MPa) \n: "))
                A = float(input("Kesit alanı giriniz. (m2) \n: "))
                print(" ")
                print(" ")
                    
                for i in dugum_listesi:
                    if aa == i.numara:
                        a_indeks = dugum_listesi.index(i)
                        if bb == i.numara:
                            b_indeks = dugum_listesi.index(i)
        
                uzay_cubuk_olusturucu(j+1, dugum_listesi[a_indeks], dugum_listesi[b_indeks], E, A)
                j +=1  
    



    elif sistem_datası == "Çerçeve":
        
        if c == 1:
            E = float(input("E Elastisite modülünü giriniz. (MPa) \n: "))
            A = float(input("A Kesit alanı giriniz. (m2) \n: "))
            I = float(input("I Atalet momentini giriniz. (m4) \n: "))
                    
            while j < cubuk_adedi:
                print(" ")
                print(" ")

                print(j+1,". çubuk eleman için")
                aa = int(input("Yakın düğüm noktasının numarasını giriniz. \n: "))
                bb = int(input("Uzak düğüm noktasının numarasını giriniz. \n: "))
                print(" ")
                print(" ")

                for i in dugum_listesi:
                    if aa == i.numara:
                        a_indeks = dugum_listesi.index(i)
                    if bb == i.numara:
                        b_indeks = dugum_listesi.index(i)
        
                cerceve_cubuk_olusturucu(j+1, dugum_listesi[a_indeks], dugum_listesi[b_indeks], E,I, A)
                j +=1
                
                    
        elif c == 0:
            while j < cubuk_adedi:
                print(" ")
                print(" ")

                print(j+1,". çubuk eleman için")
                aa = int(input("Yakın düğüm noktasının numarasını giriniz. \n: "))
                bb = int(input("Uzak düğüm noktasının numarasını giriniz. \n: "))
                E = float(input("E Elastisite modülünü giriniz. (MPa) \n: "))
                A = float(input("A Kesit alanı giriniz. (m2) \n: "))
                I = float(input("I Atalet momentini giriniz. (m4) \n: "))
                print(" ")
                print(" ")
                    
                for i in dugum_listesi:
                    if aa == i.numara:
                        a_indeks = dugum_listesi.index(i)
                        if bb == i.numara:
                            b_indeks = dugum_listesi.index(i)
        
                uzay_cubuk_olusturucu(j+1, dugum_listesi[a_indeks], dugum_listesi[b_indeks], E, A)
                j +=1  

    
def yuk_datası_sor(dugum_listesi , sistem_datası , cerceve_cubuk_listesi):
    
    j=0
    print(" ")
    print(" ")
    print("Yük girişi aşamasındasınız.")
    
    if sistem_datası == "Kafes":
        a = int(input("Kaç farklı düğüm noktasına yük etkiyor? \n: "))       
        while j < a:

            b = int(input("Yükün uygulandığı düğüm noktası numarası kaçtır? \n: "))
            c = float(input("Fx (KN) = "))
            d = float(input("Fy (KN) = "))
            for i in dugum_listesi:
                if b == i.numara:
                    dx = i.d1x
                    dy = i.d2y
                    
            yuk_olusturucu(dx, c)
            yuk_olusturucu(dy, d)
            j+=1
    elif sistem_datası == "Uzay Kafes":
        a = int(input("Kaç farklı düğüm noktasına yük etkiyor? \n: "))       
        while j < a:
            b = int(input("Yükün uygulandığı düğüm noktası numarası kaçtır? \n: "))
            c = float(input("Fx (KN) = "))
            d = float(input("Fy (KN) = "))
            e = float(input("Fz (KN) = "))
            for i in dugum_listesi:
                if b == i.numara:
                    dx = i.d1x
                    dy = i.d2y
                    dz = i.d3z
                    
            yuk_olusturucu(dx, c)
            yuk_olusturucu(dy, d)
            yuk_olusturucu(dz, e)
            j+=1      
            
    elif sistem_datası == "Çerçeve":
        
        s = int(input("Tekil Yük var mı? Evet için 1 Hayır için 0 \n: "))
        if s == 1 :
            a = int(input("Kaç farklı düğüm noktasına yük etkiyor? \n: ")) 
            while j < a:
                b = int(input("Yükün uygulandığı düğüm noktası numarası kaçtır? \n: "))
                c = float(input("Fx (KN) = "))
                d = float(input("Fy (KN) = "))
                e = float(input("Mz (KNm) = "))
                for i in dugum_listesi:
                    if b == i.numara:
                        dx = i.d1x
                        dy = i.d2y
                        dz = i.d3z
                        
                yuk_olusturucu(dx, c)
                yuk_olusturucu(dy, d)
                yuk_olusturucu(dz, e)
                j+=1      
        s2 = int(input("Yayılı Yük var mı? Evet için 1 Hayır için 0 \n: "))
        
        if  s2 == 1:
            a = int(input("Kaç adet yayılı yük var? \n: "))
            sayac = 0
            
            while sayac < a:
                
                b1 = int(input("Başlangıç düğüm nokta numarası kaçtır? \n: "))
                b2 = int(input("Bitiş düğüm nokta numarası kaçtır? \n: "))
                w = float(input("w = (KN/m) \n: "))
                
                for i in dugum_listesi:
                    if b1 == i.numara:
                        dxN = i.d1x
                        dyN = i.d2y
                        dzN = i.d3z
                        xN = i.x
                        yN = i.y
                        nameN = i.adi
                for i in dugum_listesi:
                    if b2 == i.numara:
                        dxF = i.d1x
                        dyF = i.d2y
                        dzF = i.d3z
                        xF = i.x  
                        yF = i.y
                        nameF = i.adi           
                if yN == yF:
                        
                    L = math.sqrt(((xF-xN)**2 + (yF-yN)**2))
                    FNx = 0
                    FFx = 0
                    FNy = w*L/2  
                    FFy = w*L/2
                    FNz = (w*L**2) / 12
                    FFz = (w*L**2) / 12  
        
                    q0 = np.array([
                        [FNx],
                        [FNy],
                        [FNz],
                        [FFx],
                        [FFy],
                        [-FFz]])
         
                    if nameN == "DN":
                        yuk_olusturucu(dxN , 0)
                        yuk_olusturucu(dyN , -FNy)
                        yuk_olusturucu(dzN , -FNz)
        
                    if nameF == "DN":
                        yuk_olusturucu(dxF , 0)
                        yuk_olusturucu(dyF , -FFy)
                        yuk_olusturucu(dzF , -FFz)
        
                    for i in cerceve_cubuk_listesi:
                        if dxN == i.dNx and dxF == i.dFx:
                            i.q = i.q + q0
                else:
                    L = math.sqrt(((xF-xN)**2 + (yF-yN)**2))
                    FNy = 0
                    FFy = 0
                    FNx = w*L/2  
                    FFx = w*L/2
                    FNz = (w*L**2) / 12
                    FFz = (w*L**2) / 12  
        
                    q0 = np.array([
                        [FNx],
                        [FNy],
                        [FNz],
                        [FFx],
                        [FFy],
                        [-FFz]])
         
                    if nameN == "cerceve_dugum_noktasi":
                        yuk_olusturucu(dxN , -FNx)
                        yuk_olusturucu(dyN , -FNy)
                        yuk_olusturucu(dzN , -FNz)
                    else:
                        yuk_olusturucu(dxN , 0)
                        yuk_olusturucu(dyN , 0)
                        yuk_olusturucu(dzN , 0)     
        
                    if nameF == "cerceve_dugum_noktasi":
                        yuk_olusturucu(dxF , -FFx)
                        yuk_olusturucu(dyF , -FFy)
                        yuk_olusturucu(dzF , -FFz)
                    else:
                        yuk_olusturucu(dxF , 0)
                        yuk_olusturucu(dyF , 0)
                        yuk_olusturucu(dzF , 0)         
        
                    for i in cerceve_cubuklar:
                        if dxN == i.dNx and dxF == i.dFx:
                            i.q += q0    
                sayac += 1


##################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################



print("Rijitlik Metotu ile Yapı Çözümleme Programına hoşgeldiniz.")
sistem_ne  = soru_sor()
print(" ")


Deplasman_buyutme_katsayisi = deplasman_buyutme_katsayisi()
print(" ")
mesnet_haric_dugum_sayisi = dugum_sayisi_sorusu()
print(" ")
dugum_data_sor(i, sistem_ne, mesnet_haric_dugum_sayisi)
print(" ")


mesnet_sayisi = mesnet_sayisi_sorusu()
print(" ")
dugum_sayisi = mesnet_sayisi + mesnet_haric_dugum_sayisi
print(" ")


tutulu_sayisi = mesnet_data_sor(mesnet_haric_dugum_sayisi , sistem_ne , mesnet_sayisi , dugum_sayisi)
print(" ")


mafsal_sayisi = 0
eleman_sayisi = eleman_sayisi_sorusu()
print(" ")



if sistem_ne == "Kafes":
    
    cubuk_data_sor(eleman_sayisi, sistem_ne, kafes_dugum_listesi)
    yuk_datası_sor(kafes_dugum_listesi , sistem_ne , cerceve_cubuklar)
    
elif sistem_ne == "Uzay Kafes":
    cubuk_data_sor(eleman_sayisi, sistem_ne, uzay_dugum_listesi )
    yuk_datası_sor(uzay_dugum_listesi , sistem_ne , cerceve_cubuklar)
    
elif sistem_ne == "Çerçeve":
    cubuk_data_sor(eleman_sayisi, sistem_ne, cerceve_dugum_listesi)
    yuk_datası_sor(cerceve_dugum_listesi , sistem_ne , cerceve_cubuklar)
    



#mesnet_cokmesi_olusturucu(12,-0.01)


############################################################
if sistem_ne == "Çerçeve":
    Cerceve_sistem = cerceve_GLOBAL(eleman_sayisi ,  dugum_sayisi ,  tutulu_sayisi , mesnet_sayisi)
    cerceve_calistirici(Cerceve_sistem ,  t0 ,  Deplasman_buyutme_katsayisi)
    
elif sistem_ne == "Kafes":
    kafes_sistem =  kafes_GLOBAL(eleman_sayisi ,  dugum_sayisi , tutulu_sayisi , mesnet_sayisi)
    kafes_calistirici(kafes_sistem , t0 , Deplasman_buyutme_katsayisi)

elif sistem_ne == "Uzay Kafes":
    uzay_sistem = uzay_GLOBAL(eleman_sayisi ,  dugum_sayisi ,  tutulu_sayisi , mesnet_sayisi)
    uzay_calistirici(uzay_sistem ,  t0 ,  Deplasman_buyutme_katsayisi)

print("Çözümleme tamamlandı. 3 adet Excel dosyası ve 1 adet Png dosyası oluşturuldu.")

i = 0
