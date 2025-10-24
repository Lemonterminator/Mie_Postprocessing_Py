#ifdef __linux__
    #include <stdint.h>
    #define __int64 int64_t
#endif

#if __cplusplus
    #define EXPORT_C extern "C"
#else
    #define EXPORT_C
#endif

#ifdef __linux__
    #define DllImport EXPORT_C
#else
    #define DllImport EXPORT_C __declspec(dllimport)
#endif

enum DWStatus					// status returned from library function calls
{
    DWSTAT_OK = 0,                                // status OK
    DWSTAT_ERROR = 1,                             // error occurred in the dll
    DWSTAT_ERROR_FILE_CANNOT_OPEN = 2,            // unable to open file
    DWSTAT_ERROR_FILE_ALREADY_IN_USE = 3,         // file already in use
    DWSTAT_ERROR_FILE_CORRUPT = 4,                // file corrupted
    DWSTAT_ERROR_NO_MEMORY_ALLOC = 5,             // memory not allocated
    DWSTAT_ERROR_CREATE_DEST_FILE = 6,            // error creating destination file (only for d7z files)
    DWSTAT_ERROR_EXTRACTING_FILE = 7,             // error extracting data (only for d7z files)
    DWSTAT_ERROR_CANNOT_OPEN_EXTRACTED_FILE = 8   // error opening extracted file (only for d7z files)
};

enum DWChannelProps	// used for DWGetChannelProps() calls
{
    DW_DATA_TYPE = 0,               // get data type
    DW_DATA_TYPE_LEN_BYTES = 1,     // get length of data type in bytes
    DW_CH_INDEX = 2,                // get channel index
    DW_CH_INDEX_LEN = 3,            // get length of channel index
    DW_CH_TYPE = 4,                 // get channel type
    DW_CH_SCALE = 5,                // get channel scale
    DW_CH_OFFSET = 6,               // get channel offset
    DW_CH_XML = 7,                  // get channel XML
    DW_CH_XML_LEN = 8,              // get length of channel XML
    DW_CH_XMLPROPS = 9,             // get channel XML properties
    DW_CH_XMLPROPS_LEN = 10,        // get length of channel XML properties
    DW_CH_CUSTOMPROPS = 11,         // get channel XML custom properties
    DW_CH_CUSTOMPROPS_COUNT = 12    // get length of channel XML custom properties
};

#pragma pack(1)
struct DWFileInfo				//structure used for data file
{ 
    double sample_rate;             // the sampling rate
    double start_store_time;        // absolute time of the start of storing (days)
    double duration;                // duration of data file (seconds)
};

#pragma pack(1)
struct DWChannel				//structure used for DeweSoft channel
{
    int index;                      //unique channel identifier  
    char name[100];                 //the name of a channel
    char unit[20];                  //the unit of a channel
    char description[200];          //the description of a channel
    unsigned int color;             //specifies the color of a channel
    int array_size;                 //length of the array channel (normal array_size = 1)
    int data_type;                  // channel data type - see data types in buffer 
};

#pragma pack(1)
struct DWEvent
{
    int event_type;                //1..start event; 2..stop event
    double time_stamp;             //relative position in seconds
    char event_text[200];
};

#pragma pack(1)
struct DWReducedValue
{
    double time_stamp;              //relative time in seconds
    double ave;
    double min;
    double max;
    double rms;
};

#pragma pack(1)
struct DWArrayInfo
{
    int index;                     //unique axis identifier  
    char name[100];                //axis name
    char unit[20];                 //axis unit
    int size;                      //length of the axis size
};

//event list
#define etStart 1
#define etStop 2
#define etTrigger 3
#define etVStart 11
#define etVStop 12
#define etKeyboard 20
#define etNotice 21
#define etVoice 22
#define etPicture 23
#define etModule 24
#define etAlarm 25
#define etCursorInfo 26
#define etAlarmLevel 27

//Storing type
#define ST_ALWAYS_FAST 0
#define ST_ALWAYS_SLOW 1
#define ST_FAST_ON_TRIGGER 2
#define ST_FAST_ON_TRIGGER_SLOW_OTH 3 

//FUNCTIONS:
DllImport enum DWStatus DWInit();
DllImport enum DWStatus DWDeInit();
DllImport int DWGetVersion();
DllImport enum DWStatus DWOpenDataFile(char* file_name, struct DWFileInfo* file_info);
DllImport enum DWStatus DWCloseDataFile();
DllImport int DWGetChannelListCount();
DllImport enum DWStatus DWGetChannelList(struct DWChannel* channel_list);
DllImport __int64 DWGetScaledSamplesCount(int ch_index);
DllImport enum DWStatus DWGetScaledSamples(int ch_index, __int64 position, int count, double* data, double* time_stamp);
DllImport int DWGetEventListCount();
DllImport enum DWStatus DWGetEventList(struct DWEvent* event_list);
DllImport enum DWStatus DWExportHeader(char* file_name);
DllImport int DWGetTextChannelListCount();
DllImport enum DWStatus DWGetTextChannelList(struct DWChannel* channel_list);
DllImport __int64 DWGetTextValuesCount(int ch_index);
DllImport enum DWStatus DWGetTextValues(int ch_index, int position, int count, char* text_values, double* time_stamp);
DllImport enum DWStatus DWGetReducedValuesCount(int ch_index, int* count, double* block_size);
DllImport enum DWStatus DWGetReducedValues(int ch_index, int position, int count, struct DWReducedValue* data);
DllImport int DWGetHeaderEntryCount();
DllImport enum DWStatus DWGetHeaderEntryList(struct DWChannel* channel_list);
DllImport enum DWStatus DWGetHeaderEntryText(int ch_index, char* text_value, int text_value_size);
DllImport int DWGetStoringType();
DllImport int DWGetArrayInfoCount(int ch_index);
DllImport enum DWStatus DWGetArrayInfoList(int ch_index, struct DWArrayInfo* array_inf_list);
DllImport enum DWStatus DWGetArrayIndexValue(int ch_index, int array_info_index, int array_value_index, char* value, int value_size);
DllImport enum DWStatus DWGetChannelListItem(int array_index, int* index, char* name, char* unit, char* description, int* color, int* array_size, int max_char_size);
DllImport enum DWStatus DWGetHeaderEntryListItem(int array_index, int* index, char* name, char* unit, char* description, int* color, int* array_size, int max_char_size);
DllImport enum DWStatus DWGetEventListItem(int event_Index, int* event_type, double* time_stamp, char* event_text, int max_char_size);
DllImport enum DWStatus DWGetReducedAveValues(int ch_index, int position, int count, double* data, double* time_stamp);
DllImport enum DWStatus DWGetReducedMinValues(int ch_index, int position, int count, double* data, double* time_stamp);
DllImport enum DWStatus DWGetReducedMaxValues(int ch_index, int position, int count, double* data, double* time_stamp);
DllImport enum DWStatus DWGetReducedRMSValues(int ch_index, int position, int count, double* data, double* time_stamp);

DllImport enum DWStatus DWGetNumReaders(int* num_readers);
//DllImport enum DWStatus DWGetChannelProps(int ch_index, enum DWChannelProps ch_prop, void* buffer, int* max_len);
DllImport enum DWStatus DWGetChannelProps(int ch_index, enum DWChannelProps ch_prop, void* buffer, int* max_len);

DllImport enum DWStatus DWGetHeaderEntryTextF(int entry_number, char* text_value, int text_value_size);
DllImport enum DWStatus DWGetHeaderEntryNameF(int entry_number, char* name, int name_size);
DllImport enum DWStatus DWGetHeaderEntryIDF(int entry_number, char* ID, int name_size);
DllImport double DWGetEventTimeF(int event_number);
DllImport enum DWStatus DWGetEventTextF(int event_number, char* text, int text_size);
DllImport int DWGetReducedDataChannelCountF();
DllImport enum DWStatus DWGetReducedDataChannelNameF(int Channel_Number, char* name, int name_size);
DllImport int DWGetReducedDataChannelIndexF(char* name);
DllImport enum DWStatus DWGetRecudedDataChannelInfoF(int Channel_Number, char* X_Axis_Units, int X_Axis_Units_size, char* Y_Axis_Units, int Y_Axis_Units_size, double* Chn_Offset, int* Channel_Length, double* ch_rate);
DllImport enum DWStatus DWGetRecudedDataF(int Channel_Number, double* X_Axis, double* Y_Axis, int position, int count);
DllImport int DWGetTriggerDataTriggerCountF();
DllImport double DWGetTriggerDataTriggerTimeF(int Trigger_Number);
DllImport enum DWStatus DWGetTriggerDataChannelNameF(int Channel_Number, char* name, int name_size);
DllImport int DWGetTriggerDataChannelIndexF(char* name);
DllImport enum DWStatus DWGetTriggerDataChannelInfoF(int Trigger_Number, int Channel_Number, char* X_Axis_Units, int X_Axis_Units_size, char* Y_Axis_Units, int Y_Axis_Units_size, double* Chn_Offset, double* Channel_Length, double* ch_rate, int* ch_type);
DllImport enum DWStatus DWGetTriggerDataF(int Trigger_Number , int Channel_Number, double* Y_Axis, double* X_Axis, double position, int count);

