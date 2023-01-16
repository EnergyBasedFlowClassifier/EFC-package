import numpy as np
cimport cython

# cython: profile=True
# cython: linetrace=True
# cython: binding=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef cantor(long long [:] x, long long [:] y):
    output = np.empty(x.shape[0], dtype='double')
    cdef double [:] output_view = output
    cdef int i 
    for i in range(x.shape[0]):
        output_view[i] = (x[i] + y[i]) * (x[i] + y[i] + 1) / 2 + y[i]
    return output

def site_freq(long long [:, :] X_view,
            double psdcounts,
            int max_bin):
            
    cdef int n_attr = X_view.shape[1]
    sitefreq = np.zeros((n_attr, max_bin), dtype='double')
    
    cdef double [:, :] sitefreq_view = sitefreq

    for i in range(n_attr):
        for aa in range(max_bin):
            sitefreq_view[i, aa] = np.sum(np.equal(X_view[:, i], aa))

    sitefreq /= X_view.shape[0]
    sitefreq = ((1 - psdcounts) * sitefreq
                + psdcounts / max_bin)

    return sitefreq


# @cython.cdivision(True)
def pair_freq(long long [:, :] X_view,
            double [:, :] sitefreq_view,
            double psdcounts,
            int max_bin):
    cdef int i, j, count, ai, aj, item, x
    cdef int n_inst = X_view.shape[0]
    cdef int n_attr = X_view.shape[1]
    pairfreq = np.zeros((n_attr, max_bin, n_attr, max_bin),
                        dtype='double')

    cdef double [:, :, :, :] pairfreq_view = pairfreq

    for i in range(n_attr):
        for j in range(n_attr):
            c = cantor(X_view[:,i],X_view[:,j])
            unique,aaIdx = np.unique(c,True)
            for x, item in enumerate(unique):
                pairfreq_view[i, X_view[aaIdx[x],i],j,X_view[aaIdx[x],j]] = np.sum(np.equal(c,item))

    pairfreq /= n_inst
    pairfreq = (1-psdcounts)*pairfreq + psdcounts/(max_bin**2)

    cdef double [:, :, :, :] pairfreq_new_view = pairfreq


    for i in range(n_attr):
        for ai in range(max_bin):
            for aj in range(max_bin):
                if (ai==aj):
                    pairfreq_new_view[i,ai,i,aj] = sitefreq_view[i,ai]
                else:
                    pairfreq_new_view[i,ai,i,aj] = 0.0
    return pairfreq


def coupling(double [:, :, :, :] pairfreq_view,
            double [:, :] sitefreq_view, 
            double psdcounts, 
            int max_bin):
    cdef int i, j, ai, aj
    cdef int n_attr = sitefreq_view.shape[0]
    corr_matrix = np.empty((n_attr * (max_bin - 1),
                            n_attr * (max_bin - 1)), dtype='double')

    cdef double [:, :] corr_matrix_view = corr_matrix


    for i in range(n_attr):
        for j in range(n_attr):
            for ai in range(max_bin - 1):
                for aj in range(max_bin - 1):
                    corr_matrix_view[i * (max_bin - 1) + ai,
                                j * (max_bin - 1) + aj] = (pairfreq_view[i, ai, j, aj]
                                                                - sitefreq_view[i, ai]
                                                                * sitefreq_view[j, aj])

    inv_corr = np.linalg.inv(corr_matrix_view)
    return np.exp(np.negative(inv_corr))


def local_fields(double [:, :] coupling_view, 
                double [:, :, :, :] pairfreq_view, 
                double [:, :] sitefreq_view, 
                double psdcounts, 
                int max_bin):
    cdef int i, ai, j, aj
    cdef int n_inst = sitefreq_view.shape[0]
    fields = np.zeros((n_inst * (max_bin - 1)), dtype='double')
    cdef double [:] fields_view = fields

    for i in range(n_inst):
        for ai in range(max_bin - 1):
            fields_view[i * (max_bin - 1) + ai] = (sitefreq_view[i, ai]
                                                    / sitefreq_view[i, max_bin - 1])
            for j in range(n_inst):
                for aj in range(max_bin - 1):
                    fields_view[i * (max_bin - 1) + ai] = fields_view[i *       (max_bin - 1) + ai] / (
                        coupling_view[i * (max_bin - 1) + ai, j * (max_bin - 1) + aj]**sitefreq_view[j, aj])

    return fields


def compute_energy(self, X):
    cdef int n_inst = X.shape[0]
    cdef int n_attr = X.shape[1]
    cdef int i, j, k, j_value, k_value
    cdef double e
    cdef int max_bin = self.max_bin

    energies = np.empty(n_inst, dtype='double')

    cdef long long [:, :] X_view = X
    cdef double [:] energies_view = energies
    cdef double [:, :] coupling_view = self.coupling_matrix_
    cdef double [:] fields_view = self.local_fields_

    for i in range(n_inst):
        e = 0
        for j in range(n_attr):
            j_value = X_view[i, j]
            if j_value != (max_bin - 1):
                for k in range(j, n_attr):
                    k_value = X_view[i, k]
                    if k_value != (max_bin - 1):
                        e -= (coupling_view[j * (max_bin - 1)
                                                    + j_value, k * (max_bin - 1) + k_value])
                e -= (fields_view[j * (max_bin - 1) + j_value])


        energies_view[i] = e
    return energies


def breakdown_energy(self, x):
    cdef int n_attr = x.shape[1]
    cdef int j, k, j_value, k_value
    cdef double e, holder
    cdef int max_bin = self.max_bin

    cdef double [:, :] coupling_view = self.coupling_matrix_
    cdef double [:] fields_view = self.local_fields_

    sample_couplings = np.zeros((n_attr, n_attr), dtype='double')
    sample_fields = np.zeros(n_attr, dtype='double')


    e = 0
    for j in range(n_attr):
        j_value = x[0, j]
        if j_value != (max_bin - 1):
            for k in range(j, n_attr):
                k_value = x[0, k]
                if k_value != (max_bin - 1):
                    holder = (coupling_view[j * (max_bin - 1)
                                                + j_value, k * (max_bin - 1) + k_value])
                    sample_couplings[j, k] = holder
                    sample_couplings[k, j] = holder
                    e -= holder
                    
                holder = (fields_view[j * (max_bin - 1) + j_value])
                e -= holder
                sample_fields[j] = holder

    return e, sample_fields, sample_couplings