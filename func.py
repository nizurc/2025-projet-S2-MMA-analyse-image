import cv2
import numpy as np

def extract_w(u, k1, k2, romega):# renvoie l'image correspondant à la fenêtre de centre (k1,k2)
    M,N = u.shape
    omegax = np.array(range(k1-romega,k1+romega+1))
    omegay = np.array(range(k2-romega,k2+romega+1))
    omegax = omegax % M
    omegay = omegay % M
    uw = u[np.ix_(omegax, omegay)]
    
    return uw

def guided_f(p,I,romega,epsilon): # IMPLEMENTATION NAIVE DU FILTRE GUIDÉ
    M,N = p.shape
    nomega = (1 + 2*romega)**2 # nombre de pixels dans la fenêtre
    a = np.zeros((M,N))
    b = np.zeros((M,N))
    for x in range(M):
        for y in range(N):
            Iw = extract_w(I, x, y, romega)
            pw = extract_w(p, x, y, romega)
            muk = np.mean(Iw)
            sigmak2 = np.var(Iw)
            pbarrek = np.mean(pw)
            a[x,y] = ((1/nomega) * np.sum(Iw*pw) - muk * pbarrek)/(sigmak2 + epsilon)
            b[x,y] = pbarrek - a[x,y] * muk

    q = np.zeros((M,N))
    for x in range(M):
        for y in range(N):
            aw = extract_w(a, x, y, romega)
            bw = extract_w(b, x, y, romega)
            q[x,y] = (1/nomega) * np.sum(aw * p[x,y] + bw)
    
    return q

def average_filter2(u, r):
    """
    Applique un filtre moyen sur une image 2D u avec une fenêtre carrée (2r+1)x(2r+1),
    en utilisant une image intégrale pour une implémentation rapide.
    Les bords sont symétrisés (padding miroir).
    """
    
    # Padding symétrique (miroir)
    padded = np.pad(u, ((r+1, r), (r+1, r)), mode='symmetric')
    
    # Calcul de l'image intégrale (somme cumulée 2D)
    integral = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    
    # Zone centrale : du coin supérieur gauche au coin inférieur droit
    A = integral[2*r+1:,2*r+1:]
    B = integral[:-2*r-1,2*r+1:]
    C = integral[2*r+1:,:-2*r-1]
    D = integral[:-2*r-1,:-2*r-1]
    
    total = A - B - C + D
    out = total / ((2*r + 1) ** 2)
    
    return out

def guided_f_fast2(p,I,romega,epsilon): # IMPLEMENTATION EFFICACE DU FILTRE GUIDÉ (MATRICE INTÉGRALE)
    M,N = p.shape
    a = np.zeros((M,N))
    b = np.zeros((M,N))

    p_av = average_filter2(p,romega)
    I_av = average_filter2(I,romega)
    Ip_av = average_filter2(I*p,romega)
    sig2_av = average_filter2((I-I_av)**2,romega)
    a = (Ip_av - p_av*I_av)/(sig2_av + epsilon)
    b = p_av - a * I_av
    a_av = average_filter2(a,romega)
    b_av = average_filter2(b,romega)

    q = I * a_av + b_av

    return q

