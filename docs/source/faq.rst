FAQ
===

.. _faq:

Frequently Asked Questions
--------------------------

.. contents::
   :local:
   :depth: 3

.. _installation-errors:

**Q: What should I do if I encounter errors during installation?**
    Despite our efforts to ensure astroCAST's compatibility across various environments including local and cloud-based
    systems, as well as popular operating systems such as Linux, MacOS, and Windows, unforeseen issues tied to specific
    hardware or software configurations might arise during the installation process.

    **S1: Omit optional packages**
        If you are trying to install astroCAST with the full functionality with `poetry install -E all` or `pip install
        astrocast[all]` test if the error is due to the optional packages. While the functionality might be restricted,
        this will enable you to run the majority of the toolkit.

    **S2: Force installation of missing package**
        A common root of installation problems is the inability of the installer (Poetry) to find compatible versions of
        packages required by astroCAST. A plausible first step to troubleshoot this issue is to force the installation of
        the required package through pip, which generally has more lenient installation criteria. For instance, you can use
        the following command to install a specific package:

        .. code-block:: bash

            conda activate <your_conda_env_name>
            pip install <missing_package_name>

        .. warning::

            Please note that using this method might lead to other problems stemming from version incompatibilities.
            Nevertheless, it serves as a viable initial approach to resolving installation errors.

    **S3: Open GitHub issue**
        If the above mentioned solution doesn't rectify the issue or if the problem persists, we encourage users to engage
        with our community for support. Initially, you can open an issue on our GitHub repository detailing the problem you
        are facing. As a temporary measure, if the issue remains unresolved, you might consider switching to a Linux
        operating system, which might bypass the existing problem until a solution is found.

.. _event-length-variability:
**Q: How do I handle events of variable length?**
    Different approaches are implemented in astroCAST to handle events of varying lengths. However, this inherently
    reduces the dimensionality of the data and may result in the loss of some event complexity that could be
    biologically relevant.

    **S1: Careful parameter selection**
        Explore and compare different approaches in astroCAST to preserve biologically relevant information while managing
        event length variability and data dimensionality.

.. _non-stochastic-noise:

**Q: How can I mitigate the impact of non-stochastic noise during analysis?**
    AstroCAST is susceptible to non-stochastic noise sources, such as variations in illumination intensity, which can
    introduce fluorescence contamination and affect the accuracy of the analysis. However, the protocol does not include
    specific methods to correct for these noise sources.

    **S1: Fine-tune background subtraction**
    Fine-tuning the parameters of the Delta module within astroCAST can help improve the quality of event
    detection. While this approach can be powerful, it inherently creates risk of creating artifacts by removing or
    adding spurious signals. Thoroughly validate your data after background subtraction.

    **S2: Improve imaging quality**
    Consider exploring improvements to the imaging modalities used in the experiments. Enhancements in imaging
    techniques, such as addressing illumination fluctuations, may be necessary to minimize the impact of non-stochastic
    noise. Ensuring even illumination and excluding noise pollution from other light sources is imperative. Air bubbles
    passing between the sample and the objective are another common source of non-stochastic noise, which can be
    addressed by pre-heating the feeder solution.

.. _napari-opencv-conflict:

**Q: What if I experience issues displaying videos due to napari and openCV conflicts?**
    AstroCAST uses the napari package internally to visualize data. Napari has a known conflict with openCV and problems
    can arise if both are loaded into the python kernel at the same time.

    **S1: Restart the kernel**
    This problem frequently arises when visualizing data after performing motion correction. Simply restarting the
    python kernel after motion correction (MC) and skipping the MC step should solve the problem. Alternatively, use
    another image viewer of your choice (e.g., ImageJ).

.. _event-merging:

**Q: How to address the issue of promiscuous event merging in astroCAST?**
    During the detection and extraction step, close-by events can mistakenly be recognized as a single event,
    leading to event merging.

    **S1: Adjust parameters of Delta module (if used)**
    Adjust the window length to minimize fluctuations in the baseline. If event merging occurs primarily towards the
    end of the recording, consider using the 'dFF' method, which better accounts for bleaching.

    **S2: Adjust parameters of Detection module**
    AstroCAST offers the 'split_events' functionality to split merged events based on the detection of local maxima.
    Increase the min_size or binary_struct_iterations parameter to remove noisy sections and smaller events from the
    analysis. In some cases, setting a user-defined threshold may yield better results. The detection class includes a
    "save_activepixels" parameter, allowing the export of the spatially filtered event map after thresholding for
    troubleshooting purposes.

    **S3: Improve imaging conditions**
    If fine-tuning parameters do not suffice, enhancing the recording quality through brighter calcium sensors, faster
    recording, or higher magnification may further improve results.

.. _empty-denoiser-output:

**Q: What if the denoising process results in an empty output?**
    Sometimes applying the denoiser to images results in an empty output (all values are 0).

    **S1: Check data type of input**
    During testing of astroCAST this problem occurred when the input to the denoiser had a non-standard data type.
    Ensure that you are providing input created with the :class:`astrocast.preparation.Input` module or ensure that the
    file type is float32.

.. _faulty-caching:

**Q: How to resolve issues with faulty caching in astroCAST?**
    Functions that have implemented automatic caching should detect changes to the input parameters and recalculate
    the required value. However, this automatic detection might fail in edge cases.

    **S1: Prevent caching**
    Currently the only solution is to manually delete the cache (enable logging via `logging_level=0` to print the name
    of the loaded cache file) or disable caching altogether (`cache_path=None`). If you encounter this issue please
    post an issue on the GitHub repository.

.. _missing-functionality:

**Q: What if astroCAST lacks certain functionalities I need?**
    AstroCAST's modular design allows for the integration of custom code. Please consider creating a Pull-Request if you
    end up implementing additional functionality and we will consider adding it to the repository.

.. _event-attribution:

**Q: How can I attribute events to individual cells in astroCAST?**
    AstroCAST is by design agnostic to individual cells, but rather treats each event as its own unique occurrence.
    While this approach is flexible and powerful, researchers might have specific experimental questions that rely on
    the attribution of events to individual cells (e.g., astrocytes in cell culture).

    **S1: Experimental Module feature**
    The experimental :class:`astrocast.clustering.Modules` class can act as a proxy for cell types, attributing events
    to a Functional Unit (consisting of one or few astrocytes) when correlation boundaries are sufficiently stringent.
    Please consult the function declaration for more information.

.. _long-runtime:

**Q: How to manage excessively long runtime in astroCAST?**
    Given a video file of equivalent size as described in the protocol, each protocol step should conclude within a few
    hours on most hardware utilized. However, especially in cases with many events (>100,000) runtime might exceed
    researcher’s patience.

    **S1: I/O performance**
    A common bottleneck is slow loading of chunked data from disks. If sufficient RAM is available, run the offending
    module with the `in_memory=True` setting to test if slow loading or writing speeds are causing this issue.

    **S2: Optimizing chunk size**
    Optimize the chunk size (parameters: `chunk_strategy` or `chunks`) used during processing. Choosing a size that is
    too small, too large or employing the wrong strategy can significantly slow down computations.
    Test different chunk sizes to find the most efficient one for your data. If necessary, rechunking data between
    steps may be required to achieve optimal performance. As a rule of thumb choose a chunk size of 10-100MB
    (depending on your RAM). Different sections of the toolkit work optimally with specific layouts, in short:
    chunked frames [`chunk_strategy='XY' for denoising, gaussian blurring, spatial thresholding],
    chunked time [`chunk_strategy='Z' for background subtraction, temporal thresholding] and
    balanced [`chunk_strategy='balanced' for default for others].

    **S3: Slow clustering**
    For clustering based on distance measurements (Step 21), where the complexity is O(n²), consider dimensionality
    reduction (Step 20) to be a mandatory step. If high event numbers make distance-based clustering impractical,
    employing a more aggressive filtering approach (e.g., excluding long events) can be a solution, although this may
    not align with experimental requirements. Alternatively, breaking up long recordings into individual sections can
    help overcome this limitation.

    AstroCAST also provides an two-step linkage approach
    (:class:`astrocast.clustering.Linkage.get_two_step_barycenters`) specifically designed for many events.
    However, this implementation is still experimental and currently not well tested.

.. _insufficient-memory:

**Q: What should I do if I face memory issues with large files in astroCAST?**
    AstroCAST is designed to be scalable across hardware, but there may be cases where insufficient memory becomes a
    challenge, especially with large file sizes and event numbers.

    **S1: Reducing input size**
    Consider reducing the size of the analyzed videos by downsampling (:class:`astrocast.preparation.Input`), cropping
    your video to include only relevant sections or analyzing your video in sections.

    **S2: Fine-tune parameters**
    Explore the `in-memory`, `lazy`, `chunk_strategy` and `chunks` parameters to see if this solves the problem.

    **S3: Aggressive embedding**
    When embedding your data in a latent space (e.g., :class:`astrocast.autoencoders.CNN_Autoencoder`) consider reducing
    the size of your latent output. For some clustering algorithms embedding the data into a latent space is mandatory,
    due to prohibitively long processing times.

    **S3: Explore other clustering algorithms**
    Experiment with different clustering algorithms provided by astroCAST. Memory requirements vary between approaches,
    and some may be more suitable for the number of events you need to analyze. Keep in mind that while astroCAST is
    designed to scale with available hardware, certain functionality may require hardware that exceeds standard consumer
    grade hardware.