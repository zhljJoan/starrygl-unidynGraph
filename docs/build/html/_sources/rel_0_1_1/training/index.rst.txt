Training Guides by Graph Mode
=============================

StarryGL provides tailored training paths for different temporal graph
semantics. The overall user workflow is straightforward:

1. Analyze your data and determine whether it is best represented as discrete
   snapshots or as a continuous event stream.
2. Choose the training mode that matches your data representation and model
   semantics.
3. Run offline preprocessing once to generate partition-aware artifacts, then
   launch the distributed training runtime.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   dtdg
   ctdg
   chunk

How To Choose
-------------

Use the following guide to choose the mode that best fits your workload:

**DTDG**
   Choose **DTDG** when your data naturally forms discrete snapshots, such as
   daily interactions or monthly roll-ups, and when your model needs to reason
   about structural topology within explicit time windows.

**CTDG**
   Choose **CTDG** when your data is a continuous, ordered event stream, such
   as timestamped financial transactions or user clicks, and when exact event
   order and fine-grained time intervals are critical.

**CHUNK (Experimental)**
   Choose **CHUNK** when you want a unified abstraction that frees you from managing distinct input data formats and pipeline configurations. Use this mode if you no longer want to worry about whether the underlying data is discrete or continuous