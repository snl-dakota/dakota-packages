!   _______________________________________________________________________
!
!   LHS (Latin Hypercube Sampling) wrappers for C clients.
!   Copyright (c) 2006, Sandia National Laboratories.
!   This software is distributed under the GNU Lesser General Public License.
!   For more information, see the README file in the LHS directory.
!
!   NOTE: this "C wrapper layer" is NOT a part of the original LHS source
!   code.  It was added by the DAKOTA team to allow C clients to easily
!   link with the LHS f90 routines without having to assume the burden
!   of managing the "mixed-language string translations" themselves.
!   _______________________________________________________________________
!
C These Fortran wrappers circumvent problems with implicit string sizes
C in f90.

C -----------------------------
C Wrapper for LHS's lhs_options
C -----------------------------
!LHS_EXPORT_DEC ATTRIBUTES DLLEXPORT::lhs_options2
      subroutine lhs_options2( lhsreps, lhspval, lhsopts, ierror )

C Fix the string size and always call lhs_options2 from C++ with strings of
C length 32
      character*32 lhsopts
      integer      lhsreps, lhspval, ierror

C Since calling from F90 now, the implicit string size passing should work
      call lhs_options( lhsreps, lhspval, lhsopts, ierror )

      end

C --------------------------
C Wrapper for LHS's lhs_dist
C --------------------------
!LHS_EXPORT_DEC ATTRIBUTES DLLEXPORT::lhs_dist2
      subroutine lhs_dist2( namvar, iptflag, ptval, distype, aprams,
     1                      numprms, ierror, idistno, ipvno )

C Fix the string size and always call lhs_dist2 from C++ with strings of
C length 32
      character*16     namvar
      character*32     distype
      integer          iptflag, numprms, ierror, idistno, ipvno
      double precision ptval, aprams(numprms) 

C Since calling from F90 now, the implicit string size passing should work
      call lhs_dist( namvar, iptflag, ptval, distype, aprams,
     1               numprms, ierror, idistno, ipvno )

      end

C ---------------------------
C Wrapper for LHS's lhs_udist
C ---------------------------
!LHS_EXPORT_DEC ATTRIBUTES DLLEXPORT::lhs_udist2
      subroutine lhs_udist2( namvar, iptflag, ptval, distype, numpts,
     1                       xval, yval, ierror, idistno, ipvno )

C Fix the string size and always call lhs_udist2 from C++ with strings of
C length 32
      character*16     namvar
      character*32     distype
      integer          iptflag, numpts, ierror, idistno, ipvno
      double precision ptval, xval(1), yval(1)

C Since calling from F90 now, the implicit string size passing should work
      call lhs_udist( namvar, iptflag, ptval, distype, numpts,
     1                xval, yval, ierror, idistno, ipvno )

      end

C ---------------------------
C Wrapper for LHS's lhs_const
C ---------------------------
!LHS_EXPORT_DEC ATTRIBUTES DLLEXPORT::lhs_const2
      subroutine lhs_const2( namvar, ptval, ierror, ipvno )

C Fix the string size and always call lhs_const2 from C++ with strings of
C length 32
      character*16     namvar
      integer          ierror, ipvno
      double precision ptval

C Since calling from F90 now, the implicit string size passing should work
      call lhs_const( namvar, ptval, ierror, ipvno )

      end

C --------------------------
C Wrapper for LHS's lhs_corr
C --------------------------
!LHS_EXPORT_DEC ATTRIBUTES DLLEXPORT::lhs_corr2
      subroutine lhs_corr2( nam1, nam2, corrval, ierror )

C Fix the string size and always call lhs_corr2 from C++ with strings of
C length 32
      character*16     nam1, nam2
      integer          ierror
      double precision corrval

C Since calling from F90 now, the implicit string size passing should work
      call lhs_corr( nam1, nam2, corrval, ierror )

      end

C --------------------------
C Wrapper for LHS's lhs_run
C --------------------------
!LHS_EXPORT_DEC ATTRIBUTES DLLEXPORT::lhs_run2
      subroutine lhs_run2( max_var, max_obs, max_names, ierror, 
     1                     dist_names, name_order, pt_vals, num_names,
     2                     sample_matrix, num_vars, rank_matrix, rflag )

      integer          max_var, max_obs, max_names, num_names, num_vars
      integer          rflag, ierror, name_order(1) 
      character*16     dist_names(1) 
      double precision pt_vals(1), sample_matrix(1),rank_matrix(1)

      call lhs_run( max_var, max_obs, max_names, ierror, 
     1              dist_names, name_order, pt_vals, num_names,
     2              sample_matrix, num_vars, rank_matrix, rflag )

      end

C ---------------------------
C Wrapper for LHS's lhs_files
C ---------------------------
!LHS_EXPORT_DEC ATTRIBUTES DLLEXPORT::lhs_files2
      subroutine lhs_files2( lhsout, lhsmsg, lhstitl, lhsopts, ierror )

C Fix the string size and always call lhs_files from C++ with strings of
C length 32
      character*32 lhsout, lhsmsg, lhstitl, lhsopts
      integer      ierror

C Since calling from F90 now, the implicit string size passing should work
      call lhs_files( lhsout, lhsmsg, lhstitl, lhsopts, ierror )
      
      end
