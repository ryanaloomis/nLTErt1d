# TODO write function to output levels


      SUBROUTINE blowpops(outfile,molfile,goal,snr,percent,stage,fixset
     $  ,trace)

c (c) Michiel Hogerheijde / Floris van der Tak 2000
c     michiel@strw.leidenuniv.nl, vdtak@sron.nl
c     http://www.sron.rug.nl/~vdtak/ratran
c
c     This file is part of the 'ratran' molecular excitation and
c     radiative transfer code. The one-dimensional version of this code
c     is publicly available; the two-dimensional version is available on
c     collaborative basis. Although the code has been thoroughly tested,
c     the authors do not claim that it is free of errors or that it gives
c     correct results in all situations. Any publication making use of
c     this code should include a reference to Hogerheijde & van der Tak,
c     2000, A&A, 362, 697.

c     For revision history see http://www.sron.rug.nl/~vdtak/ratran/

c     Output routine of AMC. Writes populations of all levels in all
c     grid points to file. Unit 13 is the output file.

      IMPLICIT NONE
      INCLUDE 'amccommon.inc'
      INTEGER id,j,length,stage
      DOUBLE PRECISION goal,snr,percent,fixset
      CHARACTER*80 outfile,molfile
      CHARACTER*42 fmt
      EXTERNAL length
      INTEGER count
      SAVE count
      CHARACTER*80 out2
      LOGICAL trace
      LOGICAL debug
      PARAMETER(debug=.false.)

c     id,j:    counters
c     length:  returns length of a string
c     goal:    requested s/n
c     snr:     acquired minimum s/n
c     percent: % cells exceeding goal
c     molfile: full path + file name molecular data
c     outfile: output file name
c     fmt:     helps in defining format
c     count:   keeps track of intermediate output files
c     out2:    intermediate output file name, defaults to prelim_nnn.pop
c     trace:   leave convergence history or not?
c     debug: turns debugging output on/off

      count=count+1
      if (((percent.lt.100.d0).or.(stage.eq.1)).and.(trace)) then
        write(out2,'(A,A1,I6.6,A4)') 
     $    outfile(1:length(outfile)),'_',count,'.his'
        open(unit=13,file=out2,status='unknown',err=911)
      else
        open(unit=13,file=outfile,status='unknown',err=911)
      endif
      if (debug) print*,'[debug] opened output file'

c     Write header

      write(13,'(A)') '#AMC: output file'
      if (stage.eq.1) then
        write(13,'(A,1p,E10.3)') '#AMC: fixset convergence limit='
     $    ,fixset
        write(13,'(A,1p,E10.3)') '#AMC: convergence reached=',1./snr
      else
        write(13,'(A,1p,E10.3)') '#AMC: requested snr=',goal
        write(13,'(A,1p,E10.3)') '#AMC: minimum snr=',snr
      endif
      write(13,'(A,F5.1,A)') '#AMC: ',percent,'% converged'

      write(13,'(A,1p,E12.6)') 'rmax=',rmax
      if (zmax.gt.0.) write(13,'(A,1p,E12.6)') 'zmax=',zmax
      write(fmt,'(A)') '(A,I3)'
      write(13,'(A,1p,I10.10)') 'ncell=',ncell
      write(13,'(A,1p,E9.3)') 'tcmb=',tcmb
      if (gas2dust.gt.0.d0) 
     $  write(13,'(A,1p,E9.3)') 'gas:dust=',gas2dust
      if (zmax.gt.0.d0) then
        write(13,'(A)') 
     $    'columns=id,ra,rb,za,zb,nh,tk,nm,vr,vz,va,db,td,lp'
      else
        write(13,'(A)') 
     $    'columns=id,ra,rb,nh,tk,nm,vr,db,td,lp'
      endif
      write(13,'(A,A)') 'molfile=',molfile(1:length(molfile))

      write(13,'(A)') '@'
      if (debug) print*,'[debug] wrote header'


c     Write grid, reverting to 'natural' units

      if (zmax.gt.0.d0) then
        do id=1,ncell
      if (debug) print*,'[debug] writing cell ',id
          write(fmt,'(''(I6,X,1p,12(E15.6,1X),'',I3,''(E15.6,1X))'')')
     $      nlev
          if ((doppb(id)**2.d0-2.d0*kboltz/amass*tkin(id)).lt.eps) then
            write(13,fmt)
     $        id,ra(id),rb(id),za(id),zb(id),nh2(id)/1.d6,tkin(id),
     $        nmol(id)/1.d6,vr(id)/1.d3,vz(id)/1.d3,va(id)/1.d3,
     $        0.d0,tdust(id),(pops(j,id),j=1,nlev)
          else
            write(13,fmt)
     $        id,ra(id),rb(id),za(id),zb(id),nh2(id)/1.d6,tkin(id),
     $        nmol(id)/1.d6,vr(id)/1.d3,vz(id)/1.d3,va(id)/1.d3,
     $        dsqrt(doppb(id)**2.d0-2.d0*kboltz/amass*tkin(id))/1.d3,
     $        tdust(id),(pops(j,id),j=1,nlev)
          endif
        enddo
      else
        do id=1,ncell
      if (debug) print*,'[debug] writing cell ',id
          write(fmt,'(''(I6,X,1p,8(E15.6,1X),'',I3,''(E15.6,1X))'')')
     $      nlev
          if ((doppb(id)**2.d0-2.d0*kboltz/amass*tkin(id)).lt.eps) then
          write(13,fmt)
     $      id,ra(id),rb(id),nh2(id)/1.d6,tkin(id),
     $      nmol(id)/1.d6,vr(id)/1.d3,
     $      0.d0,tdust(id),(pops(j,id),j=1,nlev)
          else
          write(13,fmt)
     $      id,ra(id),rb(id),nh2(id)/1.d6,tkin(id),
     $      nmol(id)/1.d6,vr(id)/1.d3,
     $      dsqrt(doppb(id)**2.d0-2.d0*kboltz/amass*tkin(id))/1.d3,
     $      tdust(id),(pops(j,id),j=1,nlev)
          endif
        enddo
      endif


      close(13)                 ! close output file


      RETURN


  911 STOP 'AMC: error opening output file...abort'
      END
