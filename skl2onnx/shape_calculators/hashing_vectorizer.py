# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from ..common._registration import register_shape_calculator
from ..common.utils import check_input_and_output_numbers


def calculate_sklearn_hashing_vectorizer_output_shapes(operator):
    '''
    Allowed input/output patterns are
        1. Map ---> [1, C]

    C is the total number of allowed keys in the input dictionary.
    '''
    check_input_and_output_numbers(operator, input_count_range=1,
                                   output_count_range=1)

    C = operator.raw_operator.n_features
    operator.outputs[0].type.shape = [None, C]


register_shape_calculator('SklearnHashingVectorizer',
                          calculate_sklearn_hashing_vectorizer_output_shapes)
