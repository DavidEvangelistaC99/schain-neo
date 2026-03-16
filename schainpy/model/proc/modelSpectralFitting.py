import numpy

def setConstants(dataOut):
    dictionary = {}
    dictionary["M"] = dataOut.normFactor
    dictionary["N"] = dataOut.nFFTPoints
    dictionary["ippSeconds"] = dataOut.ippSeconds
    dictionary["K"] = dataOut.nIncohInt
    
    return dictionary
        
def initialValuesFunction(data_spc, constants):
    #Constants
    M = constants["M"]
    N = constants["N"]
    ippSeconds = constants["ippSeconds"]

    S1 = data_spc[0,:]/(N*M)
    S2 = data_spc[1,:]/(N*M)
    
    Nl=min(S1)
    A=sum(S1-Nl)/N 
    #x = dataOut.getVelRange() #below matches Madrigal data better
    x=numpy.linspace(-(N/2)/(N*ippSeconds),(N/2-1)/(N*ippSeconds),N)*-(6.0/2)
    v=sum(x*(S1-Nl))/sum(S1-Nl)
    al1=numpy.sqrt(sum(x**2*(S1-Nl))/sum(S2-Nl)-v**2)
    p0=[al1,A,A,v,min(S1),min(S2)]#first guess(width,amplitude,velocity,noise)
    return p0
    
def modelFunction(p, constants):
    ippSeconds = constants["ippSeconds"]
    N = constants["N"]
    
    fm_c = ACFtoSPC(p, constants)
    fm = numpy.hstack((fm_c[0],fm_c[1]))
    return fm

def errorFunction(p, constants, LT):
    
    J=makeJacobian(p, constants) 
    J =numpy.dot(LT,J)
    covm =numpy.linalg.inv(numpy.dot(J.T ,J))
    #calculate error as the square root of the covariance matrix diagonal 
    #multiplying by 1.96 would give 95% confidence interval
    err =numpy.sqrt(numpy.diag(covm))
    return err

#-----------------------------------------------------------------------------------

def ACFw(alpha,A1,A2,vd,x,N,ippSeconds):
    #creates weighted autocorrelation function based on the operational model
    #x is n or N-n    
    k=2*numpy.pi/3.0
    pdt=x*ippSeconds
    #both correlated channels ACFs are created at the sametime
    R1=A1*numpy.exp(-1j*k*vd*pdt)/numpy.sqrt(1+(alpha*k*pdt)**2)
    R2=A2*numpy.exp(-1j*k*vd*pdt)/numpy.sqrt(1+(alpha*k*pdt)**2)
    # T is the triangle weigthing function
    T=1-abs(x)/N
    Rp1=T*R1
    Rp2=T*R2
    return [Rp1,Rp2]

def ACFtoSPC(p, constants):
    #calls the create ACF function and transforms the ACF to spectra
    N = constants["N"]
    ippSeconds = constants["ippSeconds"]
    
    n=numpy.linspace(0,(N-1),N)
    Nn=N-n
    R = ACFw(p[0],p[1],p[2],p[3],n,N,ippSeconds)
    RN = ACFw(p[0],p[1],p[2],p[3],Nn,N,ippSeconds)
    Rf1=R[0]+numpy.conjugate(RN[0])
    Rf2=R[1]+numpy.conjugate(RN[1])
    sw1=numpy.fft.fft(Rf1,n=N)
    sw2=numpy.fft.fft(Rf2,n=N)
    #the fft needs to be shifted, noise added, and takes only the real part
    sw0=numpy.real(numpy.fft.fftshift(sw1))+abs(p[4])
    sw1=numpy.real(numpy.fft.fftshift(sw2))+abs(p[5])
    return [sw0,sw1]

def makeJacobian(p, constants):
    #create Jacobian matrix
    N = constants["N"]
    IPPt = constants["ippSeconds"]
    
    n=numpy.linspace(0,(N-1),N)
    Nn=N-n
    k=2*numpy.pi/3.0
    #created weighted ACF    
    R=ACFw(p[0],p[1],p[2],p[3],n,N,IPPt)
    RN=ACFw(p[0],p[1],p[2],p[3],Nn,N,IPPt)
    #take derivatives with respect to the fit parameters
    Jalpha1=R[0]*-1*(k*n*IPPt)**2*p[0]/(1+(p[0]*k*n*IPPt)**2)+numpy.conjugate(RN[0]*-1*(k*Nn*IPPt)**2*p[0]/(1+(p[0]*k*Nn*IPPt)**2))
    Jalpha2=R[1]*-1*(k*n*IPPt)**2*p[0]/(1+(p[0]*k*n*IPPt)**2)+numpy.conjugate(RN[1]*-1*(k*Nn*IPPt)**2*p[0]/(1+(p[0]*k*Nn*IPPt)**2)) 
    JA1=R[0]/p[1]+numpy.conjugate(RN[0]/p[1])
    JA2=R[1]/p[2]+numpy.conjugate(RN[1]/p[2])
    Jvd1=R[0]*-1j*k*n*IPPt+numpy.conjugate(RN[0]*-1j*k*Nn*IPPt)
    Jvd2=R[1]*-1j*k*n*IPPt+numpy.conjugate(RN[1]*-1j*k*Nn*IPPt)
    #fft
    sJalp1=numpy.fft.fft(Jalpha1,n=N)
    sJalp2=numpy.fft.fft(Jalpha2,n=N)
    sJA1=numpy.fft.fft(JA1,n=N)
    sJA2=numpy.fft.fft(JA2,n=N)
    sJvd1=numpy.fft.fft(Jvd1,n=N)
    sJvd2=numpy.fft.fft(Jvd2,n=N)
    sJalp1=numpy.real(numpy.fft.fftshift(sJalp1))
    sJalp2=numpy.real(numpy.fft.fftshift(sJalp2))
    sJA1=numpy.real(numpy.fft.fftshift(sJA1))
    sJA2=numpy.real(numpy.fft.fftshift(sJA2))
    sJvd1=numpy.real(numpy.fft.fftshift(sJvd1))
    sJvd2=numpy.real(numpy.fft.fftshift(sJvd2))
    sJnoise=numpy.ones(numpy.shape(sJvd1))
    #combine arrays
    za=numpy.zeros([N])
    sJalp=zip(sJalp1,sJalp2)
    sJA1=zip(sJA1,za)
    sJA2=zip(za,sJA2)
    sJvd=zip(sJvd1,sJvd2)
    sJn1=zip(sJnoise, za)
    sJn2=zip(za, sJnoise)
    #reshape from 2D to 1D
    sJalp=numpy.reshape(list(sJalp), [2*N])
    sJA1=numpy.reshape(list(sJA1), [2*N])
    sJA2=numpy.reshape(list(sJA2), [2*N]) 
    sJvd=numpy.reshape(list(sJvd), [2*N]) 
    sJn1=numpy.reshape(list(sJn1), [2*N])
    sJn2=numpy.reshape(list(sJn2), [2*N])
    #combine into matrix and transpose
    J=numpy.array([sJalp,sJA1,sJA2,sJvd,sJn1,sJn2])
    J=J.T
    return J
