import numpy as np
from urh.dev.native.lib.climesdr cimport *
from libc.stdlib cimport malloc, free
# noinspection PyUnresolvedReferences
from cython.view cimport array as cvarray  # needed for converting of malloc array to python array

from urh.util.Logger import logger

cdef lms_device_t *_c_device
cdef lms_stream_t stream_tx
cdef lms_stream_t stream_rx

cpdef size_t CHANNEL_TX = 0
cpdef size_t CHANNEL_RX = 0
cpdef bool IS_TX = False

cpdef set_tx(bool is_tx):
    global IS_TX
    IS_TX = <bool>is_tx

cpdef bool get_tx():
    return IS_TX

cpdef set_channel_tx(size_t channel):
    global CHANNEL_TX
    CHANNEL_TX = <size_t>channel
    return 0

cpdef set_channel_rx(size_t channel):
    global CHANNEL_RX
    CHANNEL_RX = <size_t>channel
    return 0

cpdef size_t get_channel_tx():
    return CHANNEL_TX

cpdef size_t get_channel_rx():
    return CHANNEL_RX


cpdef list get_device_list():
    """
    Obtain a list of LMS devices attached to the system
    """
    cdef lms_info_str_t *dev_list = <lms_info_str_t *> malloc(256 * sizeof(lms_info_str_t))
    result = LMS_GetDeviceList(dev_list)
    if result > 0:
        return [dev_list[i].decode('UTF-8') for i in range(0, result)]
    else:
        return []

cpdef int open(str info=None):
    """
    Opens device specified by the provided ::lms_dev_info string
    This function should be used to open a device based upon the results of LMS_GetDeviceList()
    device should be initialized to NULL 
    :return 0 on success, (-1) on failure
    """
    cdef char*c_info
    if info is None:
        c_info = NULL
    else:
        info_byte_string = info.encode('UTF-8')
        c_info = <char *> info_byte_string

    return LMS_Open(&_c_device, c_info, NULL)

cpdef int close():
    """
    Close device
    :return:  0 on success, (-1) on failure
    """
    return LMS_Close(_c_device)

cpdef int init():
    """
    Configure LMS chip with settings that make it ready for operation.
 
    This configuration differs from default LMS chip configuration which is
    described in chip datasheet. In order to load default chip configuration use LMS_Reset().
    :return: 0 on success, (-1) on failure
    """
    return LMS_Init(_c_device)


cpdef int reset():
    return LMS_Reset(_c_device)

cpdef int synchronize():
    return LMS_Synchronize(_c_device, True)

cpdef int get_num_channels():
    """
    Obtain number of RX or TX channels. Use this to determine the maximum
    channel index (specifying channel index is required by most API functions).
    The maximum channel index is N-1, where N is number returned by this function
    :return:  Number of channels on success, (-1) on failure
    """
    return LMS_GetNumChannels(_c_device, IS_TX)

cpdef int enable_channel(bool enabled, bool is_tx, size_t channel):
    """
    Enable or disable specified RX channel.
    
    :param enabled: true(1) to enable, false(0) to disable.
    :return:  0 on success, (-1) on failure
    """
    return LMS_EnableChannel(_c_device, is_tx, channel, enabled)

cpdef enable_all_channels():
    enable_channel(True, False, 0)
    enable_channel(True, False, 1)
    enable_channel(True, True, 0)
    enable_channel(True, True, 1)

cpdef disable_all_channels():
    enable_channel(False, False, 0)
    enable_channel(False, False, 1)
    enable_channel(False, True, 0)
    enable_channel(False, True, 1)

cpdef int set_sample_rate(float_type rate, size_t oversample=0):
    """
    Set sampling rate for all RX/TX channels. Sample rate is in complex samples
    (1 sample = I + Q). The function sets sampling rate that is used for data
    exchange with the host. It also allows to specify higher sampling rate to be
    used in RF by setting oversampling ratio. Valid oversampling values are 1, 2,
    4, 8, 16, 32 or 0 (use device default oversampling value).
    :param rate: sampling rate in Hz to set
    :param oversample: RF oversampling ratio
    :return:  0 on success, (-1) on failure
    """
    LMS_SetSampleRate(_c_device, rate, oversample)

cpdef tuple get_sample_rate_range():
    """
    Get the range of supported sampling rates.
    :return: Tuple (start, end, step) of Allowed sample rate range in Hz, (-1, -1, -1) on Error
    """
    cdef lms_range_t sample_rate_range
    result = LMS_GetSampleRateRange(_c_device, IS_TX, &sample_rate_range)
    if result == 0:
        return sample_rate_range.min, sample_rate_range.max, sample_rate_range.step
    else:
        return -1, -1, -1

cpdef int set_center_frequency_tx(float_type frequency):
    """
    Set RF center frequency in Hz. This automatically selects the appropriate
    antenna (band path) for the desired frequency. In order to override antenna selection use LMS_SetAntenna().
    :param frequency: Desired RF center frequency in Hz
    :return: 0 on success, (-1) on failure
    """
    return LMS_SetLOFrequency(_c_device, True, CHANNEL_TX, frequency)

cpdef int set_center_frequency_rx(float_type frequency):
    """
    Set RF center frequency in Hz. This automatically selects the appropriate
    antenna (band path) for the desired frequency. In order to override antenna selection use LMS_SetAntenna().
    :param frequency: Desired RF center frequency in Hz
    :return: 0 on success, (-1) on failure
    """
    return LMS_SetLOFrequency(_c_device, False, CHANNEL_RX, frequency)

cpdef tuple get_center_frequency_range():
    """
    Obtain the supported RF center frequency range in Hz.
    
    :return: Tuple (start, end, step) of allowed center freq range in Hz, (-1, -1, -1) on Error
    """
    cdef lms_range_t center_freq_range
    result = LMS_GetLOFrequencyRange(_c_device, IS_TX, &center_freq_range)
    if result == 0:
        return center_freq_range.min, center_freq_range.max, center_freq_range.step
    else:
        return -1, -1, -1

cpdef int set_normalized_gain_tx(float_type gain):
    """
    Set the combined gain value
    
    This function computes and sets the optimal gain values of various amplifiers
    that are present in the device based on desired normalized gain value.
    
    :param gain: Desired gain, range [0, 1.0], where 1.0 represents the maximum gain
    :return:  0 on success, (-1) on failure
    """
    return LMS_SetNormalizedGain(_c_device, True, CHANNEL_TX, gain)

cpdef int set_normalized_gain_rx(float_type gain):
    """
    Set the combined gain value

    This function computes and sets the optimal gain values of various amplifiers
    that are present in the device based on desired normalized gain value.

    :param gain: Desired gain, range [0, 1.0], where 1.0 represents the maximum gain
    :return:  0 on success, (-1) on failure
    """
    return LMS_SetNormalizedGain(_c_device, False, CHANNEL_RX, gain)

cpdef float_type get_normalized_gain(bool is_tx):
    """
    Obtain the current combined gain value
    
    :return: Current gain, range [0, 1.0], where 1.0 represents the maximum gain, or -1 on error
    """
    cdef float_type gain = 0.0
    if is_tx:
        result = LMS_GetNormalizedGain(_c_device, True, CHANNEL_TX, &gain)
    else:
        result = LMS_GetNormalizedGain(_c_device, False, CHANNEL_RX, &gain)
    if result == 0:
        return gain
    else:
        return -1

cpdef int set_lpf_bandwidth_tx(float_type bandwidth):
    """
    Configure analog LPF of the LMS chip for the desired RF bandwidth.
    This function automatically enables LPF.
    
    :param bandwidth: LPF bandwidth in Hz
    :return: 0 on success, (-1) on failure
    """
    return LMS_SetLPFBW(_c_device, True, CHANNEL_TX, bandwidth)

cpdef int set_lpf_bandwidth_rx(float_type bandwidth):
    """
    Configure analog LPF of the LMS chip for the desired RF bandwidth.
    This function automatically enables LPF.

    :param bandwidth: LPF bandwidth in Hz
    :return: 0 on success, (-1) on failure
    """
    return LMS_SetLPFBW(_c_device, False, CHANNEL_RX, bandwidth)

cpdef list get_antenna_list(bool is_tx):
    """
    Obtain antenna list with names. First item in the list is the name of antenna index 0.
    :return: 
    """
    cdef lms_name_t *ant_list = <lms_name_t *> malloc(256 * sizeof(lms_name_t))
    if is_tx:
        result = LMS_GetAntennaList(_c_device, True, CHANNEL_TX, ant_list)
    else:
        result = LMS_GetAntennaList(_c_device, False, CHANNEL_RX, ant_list)
    if result > 0:
        return [ant_list[i].decode('UTF-8') for i in range(0, result)]
    else:
        return []

cpdef int set_antenna_tx(size_t index):
    """
    Select the antenna for the specified RX or TX channel.
    
    LMS_SetFrequency() automatically selects antenna based on frequency. 
    This function is meant to override path selected by LMS_SetFrequency() and should be called after LMS_SetFrequency().
    :param index: Index of antenna to select
    :return: 0 on success, (-1) on failure
    """
    return LMS_SetAntenna(_c_device, True, CHANNEL_TX, index)

cpdef int set_antenna_rx(size_t index):
    """
    Select the antenna for the specified RX or TX channel.

    LMS_SetFrequency() automatically selects antenna based on frequency.
    This function is meant to override path selected by LMS_SetFrequency() and should be called after LMS_SetFrequency().
    :param index: Index of antenna to select
    :return: 0 on success, (-1) on failure
    """
    return LMS_SetAntenna(_c_device, False, CHANNEL_RX, index)

cpdef int get_antenna(bool is_tx):
    """
    Obtain currently selected antenna of the the specified RX or TX channel.
     
    :return: Index of selected antenna on success, (-1) on failure
    """
    if is_tx:
        return LMS_GetAntenna(_c_device, True, CHANNEL_TX)
    else:
        return LMS_GetAntenna(_c_device, False, CHANNEL_RX)

cpdef float_type get_clock_freq(size_t clk_id):
    cdef float_type clock_hz = 0.0
    result = LMS_GetClockFreq(_c_device, clk_id, &clock_hz)
    if result == 0:
        return clock_hz
    else:
        return -1

cpdef int set_clock_freq(size_t clk_id, float_type frequency_hz):
    return LMS_SetClockFreq(_c_device, clk_id, frequency_hz)

cpdef float_type get_chip_temperature():
    """
    Read LMS7 chip internal temperature sensor
    
    :return: Temperature on success, (-1) on failure
    """
    cdef float_type chip_temp = 0.0
    result = LMS_GetChipTemperature(_c_device, 0, &chip_temp)
    if result == 0:
        return chip_temp
    else:
        return -1

cpdef int setup_stream_tx(uint32_t fifo_size):
    """
    Create new stream based on parameters passed in configuration structure.
    The structure is initialized with stream handle.
    :param fifo_size: FIFO size (in samples) used by stream.
    :return: 0 on success, (-1) on failure
    """
    stream_tx.isTx = True
    stream_tx.channel = <uint32_t> CHANNEL_TX
    stream_tx.fifoSize = fifo_size
    stream_tx.dataFmt = dataFmt_t.LMS_FMT_F32
    stream_tx.throughputVsLatency = 0.0  # optimize for minimum latency

    return LMS_SetupStream(_c_device, &stream_tx)

cpdef int destroy_stream_tx():
    """
    Deallocate memory used for stream.
    :return: 0 on success, (-1) on failure
    """
    LMS_DestroyStream(_c_device, &stream_tx)

cpdef int start_stream_tx():
    """
    Start stream 
    :return: 0 on success, (-1) on failure
    """
    return LMS_StartStream(&stream_tx)

cpdef int stop_stream_tx():
    """
    Stop stream
    :return: 0 on success, (-1) on failure
    """
    return LMS_StopStream(&stream_tx)

cpdef int recv_stream_tx(connection, unsigned num_samples, unsigned timeout_ms):
    """
    Read samples from the FIFO of the specified stream.
    Sample buffer must be big enough to hold requested number of samples.
    
    :param num_samples: how many samples shall be read from streams FIFO
    :param connection: multiprocessing connection to send the received samples to
    :param timeout_ms: how long to wait for data before timing out.
    :return: 
    """
    cdef lms_stream_meta_t meta = lms_stream_meta_t(0, False, False)
    cdef float*buff = <float *> malloc(num_samples * 2 * sizeof(float))

    if not buff:
        raise MemoryError()

    cdef int received_samples = LMS_RecvStream(&stream_tx, buff, num_samples, &meta, timeout_ms)

    if received_samples > 0:
        connection.send_bytes(<float[:2*received_samples]>buff)
    else:
        logger.warning("LimeSDR: Failed to receive stream")

    free(buff)

cpdef int send_stream_tx(float[::1] samples, unsigned timeout_ms):
    """
    Write samples to the FIFO of the specified stream.
    
    :param samples: sample buffer
    :param timeout_ms: how long to wait for data before timing out
    :return: number of samples send on success, (-1) on failure
    """
    cdef lms_stream_meta_t meta = lms_stream_meta_t(0, False, False)
    if len(samples) == 1:
        samples = np.zeros(1020, dtype=np.float32)
    cdef size_t sample_count = len(samples) // 2

    if len(samples) > 0:
        return LMS_SendStream(&stream_tx, &samples[0], sample_count, &meta, timeout_ms)
    else:
        return -1

cpdef int setup_stream_rx(uint32_t fifo_size):
    """
    Create new stream based on parameters passed in configuration structure.
    The structure is initialized with stream handle.
    :param fifo_size: FIFO size (in samples) used by stream.
    :return: 0 on success, (-1) on failure
    """
    stream_rx.isTx = False
    stream_rx.channel = <uint32_t> CHANNEL_RX
    stream_rx.fifoSize = fifo_size
    stream_rx.dataFmt = dataFmt_t.LMS_FMT_F32
    stream_rx.throughputVsLatency = 0.0  # optimize for minimum latency

    return LMS_SetupStream(_c_device, &stream_rx)

cpdef int destroy_stream_rx():
    """
    Deallocate memory used for stream.
    :return: 0 on success, (-1) on failure
    """
    LMS_DestroyStream(_c_device, &stream_rx)

cpdef int start_stream_rx():
    """
    Start stream
    :return: 0 on success, (-1) on failure
    """
    return LMS_StartStream(&stream_rx)

cpdef int stop_stream_rx():
    """
    Stop stream
    :return: 0 on success, (-1) on failure
    """
    return LMS_StopStream(&stream_rx)

cpdef int recv_stream_rx(connection, unsigned num_samples, unsigned timeout_ms):
    """
    Read samples from the FIFO of the specified stream.
    Sample buffer must be big enough to hold requested number of samples.

    :param num_samples: how many samples shall be read from streams FIFO
    :param connection: multiprocessing connection to send the received samples to
    :param timeout_ms: how long to wait for data before timing out.
    :return:
    """
    cdef lms_stream_meta_t meta = lms_stream_meta_t(0, False, False)
    cdef float*buff = <float *> malloc(num_samples * 2 * sizeof(float))

    if not buff:
        raise MemoryError()

    cdef int received_samples = LMS_RecvStream(&stream_rx, buff, num_samples, &meta, timeout_ms)

    if received_samples > 0:
        connection.send_bytes(<float[:2*received_samples]>buff)
    else:
        logger.warning("LimeSDR: Failed to receive stream")

    free(buff)

cpdef int send_stream_rx(float[::1] samples, unsigned timeout_ms):
    """
    Write samples to the FIFO of the specified stream.

    :param samples: sample buffer
    :param timeout_ms: how long to wait for data before timing out
    :return: number of samples send on success, (-1) on failure
    """
    cdef lms_stream_meta_t meta = lms_stream_meta_t(0, False, False)
    if len(samples) == 1:
        samples = np.zeros(1020, dtype=np.float32)
    cdef size_t sample_count = len(samples) // 2

    if len(samples) > 0:
        return LMS_SendStream(&stream_rx, &samples[0], sample_count, &meta, timeout_ms)
    else:
        return -1

cpdef load_config(filename):
    filename_byte_string = filename.encode('UTF-8')
    c_filename = <char *> filename_byte_string
    LMS_LoadConfig(_c_device, c_filename)

cpdef save_config(str filename):
    filename_byte_string = filename.encode('UTF-8')
    c_filename = <char *> filename_byte_string
    LMS_SaveConfig(_c_device, c_filename)

cpdef void print_last_error():
    cdef char * error_msg = <char *> malloc(2000 * sizeof(char))
    error_msg = <char *>LMS_GetLastErrorMessage()
    error_msg_py = error_msg.decode("UTF-8")
    print(error_msg_py)
