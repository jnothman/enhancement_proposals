.. _slep_005:

=============
Resampler API
=============

:Author: Oliver Rausch (oliverrausch99@gmail.com),
         Christos Aridas (char@upatras.gr),
         Guillaume Lemaitre (g.lemaitre58@gmail.com)
:Status: Draft
:Created: created on, in 2019-03-01
:Resolution: <url>

Abstract
--------

We propose the inclusion of a new type of estimator: resampler. The
resampler will change the samples in ``X`` and ``y`` and return both
``Xt`` and ``yt``, as well as any other keyword arguments, and applies only
to the training set.

In short:

* a new verb/method that all resamplers must implement is introduced:
  ``fit_resample``.
* resamplers are able to reduce and/or augment the number of samples in
  ``X`` and ``y`` during ``fit``, but will perform no changes during
  ``predict``. While ``fit_transform`` is required to maintain direct
  correspondence between each input and output sample, ``fit_resample`` does
  not need its input and output to have sample-wise correspondence.
* to facilitate this behavior a new meta-estimator (``ResampledTrainer``) that
  allows for the composition of resamplers and estimators is proposed.
  Alternatively, changes to ``Pipeline`` could facilitate similar compositions.


Motivation
----------

Sample reduction or augmentation are common parts of machine-learning
pipelines. The current scikit-learn API does not offer support for such
use cases.

Possible Use Cases
..................

* Sample rebalancing to correct bias toward a class with large cardinality.
* Outlier rejection to fit a clean dataset.
* Sample reduction e.g. representing a dataset by its k-means centroids.
* NaNRejector (drop all samples that contain nan).
* Dataset augmentation (like is commonly done in deep learning).
* Currently semi-supervised learning is not supported by scoring-based
  functions like ``cross_val_score``, ``GridSearchCV`` or ``validation_curve``
  since the scorers will regard "unlabeled" as a separate class. A resampler
  could add the unlabeled samples to the dataset during fit time to solve this
  (note that this could also be solved by a new cv splitter).

Implementation
--------------

API and Constraints
...................

* Resamplers implement a method ``fit_resample(X, y, **kwargs)``, a pure
  function which returns ``Xt, yt, kwargs`` corresponding to the resampled
  dataset, where returned samples do not necessarily correspond to input
  samples.
* An estimator may only implement either ``fit_transform`` or ``fit_resample``
  if support for ``Resamplers`` in ``Pipeline`` is enabled
  (see Sect. "Limitations").
* While they are free to reorder samples, resamplers may not change the order,
  meaning, dtype or format of features (this is left to transformers).
* Resamplers should also handle (e.g. resample, generate anew, etc.) any
  kwargs.

Composition
-----------

A key part of the proposal is the introduction of a way of composing resamplers
with predictors. We present two options: ``ResampledTrainer`` and modifications
to ``Pipeline``.

Alternative 1: ResampledTrainer
...............................

This metaestimator composes a resampler and a predictor. It
behaves as follows:

* ``fit(X, y, **kw)``: resample ``X, y, kw`` with the resampler, then fit the
  predictor on the resampled dataset.
* ``predict(X)``: simply predict on ``X`` with the predictor.
* ``score(X)``: simply score on ``X`` with the predictor.

See :issue:`#13269` for an implementation.

The ``ResampledTrainer`` may not be as intuitive for usage as extending
``Pipeline`` to support resampling, but it is unambiguous as to which
estimators and contexts will receive resampled data, and which not.
Its implementation and testing is straightforward, unlike a new feature
addition to the already-complicated ``Pipeline`` implementation.

There are complications around supporting ``fit_transform``, ``fit_predict``
and ``fit_resample`` methods in ``ResampledTrainer``. ``fit_transform`` support
is only possible by implementing ``fit_transform(X, y)`` as ``fit(X,
y).transform(X)``, rather than calling ``fit_transform`` of the predictor.
``fit_predict`` would have to behave similarly.  Thus ``ResampledTrainer``
would not work with non-inductive estimators (TSNE, AgglomerativeClustering,
etc.) as their final step.  If the predictor of a ``ResampledTrainer`` is
itself a resampler, it's unclear how ``ResampledTrainer.fit_resample`` should
behave. In short, only plain ``fit`` should be executed on the predictor.
These caveats also apply to the Pipeline modification below.

One benefit of the ``ResampledTrainer`` is that it does not stop the resampler
having other methods, such as ``transform``, as it is clear that the
``ResampledTrainer`` will only call ``fit_resample``.

Example Usage:
~~~~~~~~~~~~~~

.. code-block:: python

    est = ResampledTrainer(RandomUnderSampler(), SVC())
    est = make_pipeline(
        StandardScaler(),
        ResampledTrainer(Birch(), make_pipeline(SelectKBest(), SVC()))
    )
    est = ResampledTrainer(
        RandomUnderSampler(),
        make_pipeline(StandardScaler(), SelectKBest(), SVC()),
    )
    clf = ResampledTrainer(
        NaNRejector(), # removes samples containing NaN
        ResampledTrainer(RandomUnderSampler(),
            make_pipeline(StandardScaler(), SGDClassifier()))
    )

Alternative 2: Prediction Pipeline
..................................

As an alternative to ``ResampledTrainer``, ``Pipeline`` can be modified to
accomodate resamplers.  The essence of the operation is this: one or more steps
of the pipeline may be a resampler. When fitting the Pipeline, ``fit_resample``
will be called on each resampler instead of ``fit_transform``, and the output
of ``fit_resample`` will be used in place of the original ``X``, ``y``, etc.,
to fit the subsequent step (and so on).  When predicting in the Pipeline,
the resampler will act as a passthrough step.

Limitations
~~~~~~~~~~~

.. rubric:: Prohibiting ``transform`` on resamplers

It may be problematic for a resampler to provide ``transform`` if Pipelines
support resampling:

1. It is unclear what to do at test time if a resampler has a transform
   method.
2. Adding ``fit_resample`` to the API of an an existing transformer may
   drastically change its behaviour in a ``Pipeline``.

For this reason, it may be best to reject resamplers supporting ``transform``
from being used in a Pipeline.

.. rubric:: Prohibiting ``transform`` on resampling Pipelines

Providing a ``transform`` method on a Pipeline that contains a resampler
presents several problems:

1. A resampling ``Pipeline`` needs to use a special code path for
   ``fit_transform`` that would call ``fit(X, y, **kw).transform(X)`` on the
   ``Pipeline``, rather than calling ``fit_transform`` on the last step.
   Doing so would result in the transformation of the resampled data.
   Thus the effect of the resampler is not localised in terms
   of code maintenance.
2. As a result of issue 1, appending a step to the transformation ``Pipeline``
   means that the transformer which was previously last, and previously trained
   on the full dataset, will now be trained on the resampled dataset.
3. As a result of issue 1, the last step cannot be ``'passthrough'`` as in
   other transformer pipelines.

A resampler changes the semantics of a Pipeline, and arguably makes it
ambiguous to the user where the resampler is followed by one or more
transformers.

Example Usage:
~~~~~~~~~~~~~~

.. code-block:: python

    est = make_pipeline(RandomUnderSampler(), SVC())
    est = make_pipeline(StandardScaler(), Birch(), SelectKBest(), SVC())
    est = make_pipeline(
        RandomUnderSampler(), StandardScaler(), SelectKBest(), SVC()
    )
    est = make_pipeline(
        NaNRejector(), RandomUnderSampler(), StandardScaler(), SGDClassifer()
    )
    est.fit(X,y, sgdclassifier__sample_weight=my_weight)

Handling ``fit`` parameters
---------------------------

Sample metadata ("sample props") including weights cannot be routed to steps
downstream of a resampler in a Pipeline, unless they too are resampled. To
support this, a resampler would need to be passed all props that are required
downstream, and ``fit_resample`` should return resampled versions of them.
If a resampler does not support resampling all the fit parameters it is passed,
it should raise a TypeError.

Some ambiguity arises when the resampler both uses sample-aligned metadata to
its ``fit_resample`` method, and is capable of resampling additional metadata.
For example, if a resampler supports weighted fitting, but also returns a
resampled version of each sample-aligned property it is given, should
``fit_resample(X, y, sample_weight=sample_weight)`` result in ``sample_weight``
being resampled, consumed, or both?

Solutions:

* Require that any metadata that should be resampled must have a prefix that
  signifies that it should be resampled. The prefix may then be dropped in the
  dict of keyword values returned by ``fit_resample``.
* With the solution for SLEP006 found in :issue:`16079`: all metadata passed is
  resampled. However, the `ResampledTrainer` only   

Alternatives to Resamplers
--------------------------

Alternative: Resamplers as metaestimators
.........................................

One alternative solution would require all resamplers to be implemented as
wrappers to another estimator, rather than the decoupling of resampler and
meta-estimator assumed in ``ResampledTrainer``.

This is already feasible in Scikit-learn, but:

* it fails to make clear what the "one obvious way to do it" is for users.
* there are tricky edge cases for metaestimator implementation, especially
  when the meta-estimator can act as a classifier, regressor, transformer, etc.
  Creating a notion of resamplers lowers the bar for entry.

Alternative: sample_weight modification only
............................................

Alternatively ``sample_weight`` could be used as a way to effectively perform
resampling, including sample removal. However, the current limitations are:

* ``sample_weight`` is not available for all estimators;
* ``sample_weight`` will implement only simple resampling (only when resampling
  uses original samples);
* ``sample_weight`` needs to be passed and modified within a
  ``Pipeline``, which isn't possible without something like resamplers.

Support for changing feature space as well as sample space?
-----------------------------------------------------------

The proposal above currently disallows modifying feature space. There are use
cases where the feature and sample space should both be modified, such as in
order to load both X and y from an external resource. This could be
accommodated by allowing ``fit_resample`` to modify the feature space, and
having resamplers (optionally) implement ``transform`` which would be called by
ResampledTrainer only at test time. A decision on whether to support this usage
may be deferred as long as ResampledTrainer rejects resamplers that also have
`transform` implemented.

Support for fit_resample in existing estimators
-----------------------------------------------

Outlier detectors should be provided with ``fit_resample``, allowing them to
act as outlier removers.

Initially, we do not have justification to add ``fit_resample`` support to
metaestimators, including Pipeline, GridSearchCV, etc. These could be added
at a later point.

Current implementation
----------------------

https://github.com/scikit-learn/scikit-learn/pull/13269

Backward compatibility
----------------------

There are no backward incompatibilities with the current API.

Discussion
----------

* https://github.com/scikit-learn/scikit-learn/pull/13269

Naming
......

Alternatives to "Resampler" and ``fit_resample`` that were considered include:

* ``fit_rewrite``

Alternatrives to ``ResampledTrainer`` that were considered include:

* ``ResamplingTrainer``
* ``Resampled``
* ``WithResampling``
* ``TrainWith``

References and Footnotes
------------------------

.. [1] Each SLEP must either be explicitly labeled as placed in the public
   domain (see this SLEP as an example) or licensed under the `Open
   Publication License`_.

.. _Open Publication License: https://www.opencontent.org/openpub/


Copyright
---------

This document has been placed in the public domain. [1]_
