#!/usr/bin/env python
# coding: utf-8

# Importing Relevant Libraries
import numpy as np
import itertools


def get_all_monom(monomials):
    """
    Function to create a single list of all possible monomials
    Input:
        monomials: List of List of Monomials seperate by degree
    """
    all_monom = []
    for i in range(len(monomials)):
        all_monom += monomials[i]
    return all_monom

def construct_vector(m, i):
    """Construct the vector for x_i of length 2^m, which has form:
    A string of 2^{m-i-1} 0s followed by 2^{m-i-1} 1s, repeated
    2^m / (2*2^{m-i}) =  2^{i} times.
    NOTE: we must have 0 <= i < m."""
    return np.array(([1] * (2 ** (m-i-1)) + [0] * (2 ** (m-i-1))) * (2 ** (i)))

def generate_all_rows(m, S):
    """Generate all rows over the monomials in S, e.g. if S = {0,2}, we want to generate
    a list of four rows, namely:
    construct_vector(0) * construct_vector(2)
    construct_vector(0) * !construct_vector(2)
    !construct_vector(0) * construct_vector(2)
    !construct_vector(0) * !construct_vector(2).
    
    where construct_vector is the function right above
    We do this using recursion on S."""

    if not S:
        return [[1] * (2 ** m)]

    i, Srest = S[0], S[1:]

    # Find all the rows over Srest.
    Srest_rows = generate_all_rows(m, Srest)

    # Now, for both the representation of x_i and !x_i, return the rows multiplied by these.
    xi_row = construct_vector(m, i)
    not_xi_row = 1 - xi_row
    return [np.multiply(xi_row, row) for row in Srest_rows] + [np.multiply(not_xi_row, row) for row in Srest_rows]


def get_monomial_combinations(m,r):
    """
    Function to generate all the possible combinations of monomials possible 
    with a given m and r
    Return a List of List of monomials seperate by degree
    """
    monomials = []
    # Loop through all possible degrees
    for i in range(r + 1):
        monomials_deg_i = []
        # Find all possible combinations of degree i
        for S in itertools.combinations(range(m), i):
            monomials_deg_i.append(S)
        # Group all degree i monomials in one place
        monomials.append(monomials_deg_i)
    return monomials

def get_gen_matrix(m,r):
    """
    Function to create a Reed Muller generator matrix for a given m and r
    Followed the method described here : 
    https://en.wikipedia.org/wiki/Reed%E2%80%93Muller_code#Description_using_a_generator_matrix
    """
    monomials = get_monomial_combinations(m,r)

    gen_matrix = [np.ones(2**m,dtype=int)]
    
    # Rows corresponding to the monomials X0,X1,X2,...Xm.
    indiv_rows = []
    for s in range(m):
        indiv_rows.append(np.array(construct_vector(m,s)))
    indiv_rows = np.array(indiv_rows) 
    
    for i in range(1,r+1):
        for monom in monomials[i]:
            # Generating Rows corresponding to the monomials X0X1X2...Xk 
            wedge_product = np.ones(2**m,dtype=int)
            for elem in monom:
                wedge_product = np.logical_and(wedge_product,indiv_rows[elem])
        
            gen_matrix.append(wedge_product)
        
    gen_matrix = np.array(gen_matrix) 
    
    return gen_matrix