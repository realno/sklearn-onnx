# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import warnings
from ..common._apply_operation import apply_cast, apply_reshape
from ..common._registration import register_converter
from ..proto import onnx_proto

def convert_sklearn_hashing_vectorizer(scope, operator, container):
    """
    Converters for class
    `TfidfVectorizer <https://scikit-learn.org/stable/modules/generated/
    sklearn.feature_extraction.text.TfidfVectorizer.html>`_.
    The current implementation is a work in progress and the ONNX version
    does not produce the exact same results. The converter lets the user
    change some of its parameters.

    Additional options
    ------------------

    tokenexp: string
        The default will change to true in version 1.6.0.
        The tokenizer splits into words using this regular
        expression or the regular expression specified by
        *scikit-learn* is the value is an empty string.
        See also note below.
        Default value: None
    separators: list of separators
        These separators are used to split a string into words.
        Options *separators* is ignore if options *tokenexp* is not None.
        Default value: ``[' ', '[.]', '\\\\?', ',', ';', ':', '\\\\!']``.

    Example (from :ref:`l-example-tfidfvectorizer`):

    ::

        seps = {TfidfVectorizer: {"separators": [' ', '[.]', '\\\\?', ',', ';',
                                                 ':', '!', '\\\\(', '\\\\)',
                                                 '\\n', '\\\\"', "'", "-",
                                                 "\\\\[", "\\\\]", "@"]}}
        model_onnx = convert_sklearn(pipeline, "tfidf",
                                     initial_types=[("input", StringTensorType([None, 2]))],
                                     options=seps)

    The default regular expression of the tokenizer is ``(?u)\\\\b\\\\w\\\\w+\\\\b``
    (see `re <https://docs.python.org/3/library/re.html>`_).
    This expression may not supported by the library handling the backend.
    `onnxruntime <https://github.com/Microsoft/onnxruntime>`_ uses
    `re2 <https://github.com/google/re2>`_. You may need to switch
    to a custom tokenizer based on
    `python wrapper for re2 <https://pypi.org/project/re2/>_`
    or its sources `pyre2 <https://github.com/facebook/pyre2>`_
    (`syntax <https://github.com/google/re2/blob/master/doc/syntax.txt>`_).
    If the regular expression is not specified and if
    the instance of TfidfVectorizer is using the default
    pattern ``(?u)\\\\b\\\\w\\\\w+\\\\b``, it is replaced by
    ``[a-zA-Z0-9_]+``. Any other case has to be
    manually handled.

    Regular expression ``[^\\\\\\\\n]`` is used to split
    a sentance into character (and not works) if ``analyser=='char'``.
    The mode ``analyser=='char_wb'`` is not implemented.
    
    .. versionchanged:: 1.6
        Parameters have been renamed: *sep* into *separators*,
        *regex* into *tokenexp*.
    ````
    
    """ # noqa

    op = operator.raw_operator

    if op.analyzer == "char_wb":
        raise NotImplementedError(
            "CountVectorizer cannot be converted, "
            "only tokenizer='word' is supported. "
            "You may raise an issue at "
            "https://github.com/onnx/sklearn-onnx/issues.")
    if op.strip_accents is not None:
        raise NotImplementedError(
            "CountVectorizer cannot be converted, "
            "only stip_accents=None is supported. "
            "You may raise an issue at "
            "https://github.com/onnx/sklearn-onnx/issues.")

    options = container.get_options(
            op, dict(separators="DEFAULT",
                     tokenexp=None))
    if set(options) != {'separators', 'tokenexp'}:
        raise RuntimeError("Unknown option {} for {}".format(
                                set(options) - {'separators'}, type(op)))

    if op.analyzer == 'word':
        default_pattern = '(?u)\\b\\w\\w+\\b'
        if options['separators'] == "DEFAULT" and options['tokenexp'] is None:
            warnings.warn("Converter for HashingVectorizer will use "
                          "scikit-learn regular expression by default "
                          "in version 1.6.",
                          UserWarning)
            default_separators = [' ', '.', '\\?', ',', ';', ':', '\\!']
            regex = op.token_pattern
            if regex == default_pattern:
                regex = '[a-zA-Z0-9_]+'
            default_separators = None
        elif options['tokenexp'] is not None:
            if options['tokenexp']:
                regex = options['tokenexp']
            else:
                regex = op.token_pattern
                if regex == default_pattern:
                    regex = '[a-zA-Z0-9_]+'
            default_separators = None
        else:
            regex = None
            default_separators = options['separators']
    else:
        if options['separators'] != 'DEFAULT':
            raise RuntimeError("Option separators has no effect "
                               "if analyser != 'word'.")
        regex = options['tokenexp'] if options['tokenexp'] else '.'
        default_separators = None

    if op.preprocessor is not None:
        raise NotImplementedError(
            "Custom preprocessor cannot be converted into ONNX. "
            "You may raise an issue at "
            "https://github.com/onnx/sklearn-onnx/issues.")
    if op.tokenizer is not None:
        raise NotImplementedError(
            "Custom tokenizer cannot be converted into ONNX. "
            "You may raise an issue at "
            "https://github.com/onnx/sklearn-onnx/issues.")
    if op.strip_accents is not None:
        raise NotImplementedError(
            "Operator StringNormalizer cannot remove accents. "
            "You may raise an issue at "
            "https://github.com/onnx/sklearn-onnx/issues.")

    if op.lowercase or op.stop_words:

        if len(operator.input_full_names) != 1:
            raise RuntimeError("Only one input is allowed, found {}.".format(
                operator.input_full_names))
        flatten = scope.get_unique_variable_name('flattened')
        apply_reshape(scope, operator.input_full_names[0],
                      flatten, container,
                      desired_shape=(-1, ))

        # StringNormalizer
        op_type = 'StringNormalizer'
        attrs = {'name': scope.get_unique_operator_name(op_type)}
        normalized = scope.get_unique_variable_name('normalized')
        if container.target_opset >= 10:
            attrs.update({
                'case_change_action': 'LOWER',
                'is_case_sensitive': not op.lowercase,
            })
            op_version = 10
            domain = ''
        else:
            attrs.update({
                'casechangeaction': 'LOWER',
                'is_case_sensitive': not op.lowercase,
            })
            op_version = 9
            domain = 'com.microsoft'

        if op.stop_words:
            attrs['stopwords'] = list(sorted(op.stop_words))
        container.add_node(op_type, flatten,
                           normalized, op_version=op_version,
                           op_domain=domain, **attrs)
    else:
        normalized = operator.input_full_names

    # Tokenizer
    padvalue = "#"

    op_type = 'Tokenizer'
    attrs = {'name': scope.get_unique_operator_name(op_type)}
    attrs.update({
        'pad_value': padvalue,
        'mark': False,
        'mincharnum': 1,
    })
    if regex is None:
        attrs['separators'] = default_separators
    else:
        attrs['tokenexp'] = regex

    tokenized = scope.get_unique_variable_name('tokenized')
    container.add_node(op_type, normalized, tokenized,
                       op_domain='com.microsoft', **attrs)

    # Flatten
    # Tokenizer outputs shape {1, C} or {1, 1, C}.
    # Second shape is not allowed by TfIdfVectorizer.
    # We use Flatten which produces {1, C} in both cases.
    flatt_tokenized = scope.get_unique_variable_name('flattened')
    container.add_node("Flatten", tokenized, flatt_tokenized,
                       name=scope.get_unique_operator_name('Flatten'))
    tokenized = flatt_tokenized
    mode = 'TF'

    # Create the node.

    # Set default and max n_features=5000
    if op.n_features is None:
        op.n_features = 1048576

    attrs = {'name': scope.get_unique_operator_name("HashingVectorizer")}
    attrs.update({
        'n_features': op.n_features,
    })
    output = (scope.get_unique_variable_name('output')
              if op.binary else operator.output_full_names)

    if container.proto_dtype == onnx_proto.TensorProto.DOUBLE:
        output_tf = scope.get_unique_variable_name('cast_result')
    else:
        output_tf = output

    if container.target_opset < 9:
        op_type = 'Ngram'
        container.add_node(op_type, tokenized, output_tf,
                           op_domain='com.microsoft', **attrs)
    else:
        op_type = 'HashingVectorizer'
        container.add_node(op_type, tokenized, output_tf,
                           op_domain='com.microsoft', **attrs)

    if container.proto_dtype == onnx_proto.TensorProto.DOUBLE:
        apply_cast(scope, output_tf, output,
                   container, to=container.proto_dtype)

    if op.binary:
        cast_result_name = scope.get_unique_variable_name('cast_result')

        apply_cast(scope, output, cast_result_name, container,
                   to=onnx_proto.TensorProto.BOOL)
        apply_cast(scope, cast_result_name, operator.output_full_names,
                   container, to=onnx_proto.TensorProto.FLOAT)


register_converter('SklearnHashingVectorizer', convert_sklearn_hashing_vectorizer,
                   options={'tokenexp': None, 'separators': None})
