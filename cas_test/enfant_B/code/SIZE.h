CBOP
C    !ROUTINE: SIZE.h
C    enfant du cas test (118x76x32, dx=200 m)
CEOP
      INTEGER sNx, sNy, OLx, OLy, nSx, nSy, nPx, nPy, Nx, Ny, Nr
      PARAMETER (
     &           sNx = 59,
     &           sNy = 76,
     &           OLx =   4,
     &           OLy =   4,
     &           nSx =   2,
     &           nSy =   1,
     &           nPx =   1,
     &           nPy =   1,
     &           Nx  = sNx*nSx*nPx,
     &           Ny  = sNy*nSy*nPy,
     &           Nr  = 32)
      INTEGER MAX_OLX, MAX_OLY
      PARAMETER ( MAX_OLX = OLx, MAX_OLY = OLy )
