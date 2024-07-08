import numpy
import pandas
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
import warnings
import copy
from CosinorPy import cosinor
from matplotlib.lines import Line2D
plt.ion()

def positive_rad(rad):
    """
    Convert a radian value to a positive value between 0 and 2pi

    :param rad: Radian value
    :type rad: float
    :return: Positive radian value
    :rtype: float
    """
    if rad < 0:
        return 2 * numpy.pi + rad
    else:
        return rad

def colect_data_per_day(protocol, days_to_save = 'all', save_folder = None, save_suffix = ''):
    """
    Colect data per day

    :param protocol: The protocol to colect the data per day
    :type protocol: protocol
    :param days_to_save: The days to save the data, defaults to 'all'
    :type days_to_save: list
    :param save_folder: The folder to save the data, defaults to None
    :type save_folder: str
    :param save_suffix: The suffix to add to the save file, defaults to ''
    :type save_suffix: str
    """
    protocol_df = protocol.data.copy()
    protocol_name = protocol.name.replace('_', ' ').capitalize()

    if days_to_save == 'all':
        days_to_save = protocol_df['day'].unique()
    else:
        if not isinstance(days_to_save, list):
            raise ValueError("days_to_save must be a list")
        else:
            # check all elements are int
            if isinstance(days_to_save[0], int):
                days_to_save = [int(day - 1) for day in days_to_save]
                days_in_experiment = protocol_df['day'].unique()
                days_to_save = [days_in_experiment[day] for day in days_to_save]
            elif isinstance(days_to_save[0], str):
                days_to_save = [str(day) for day in days_to_save]
            else:
                raise ValueError("The elements of days_to_save must be int or str")
        
            for day in days_to_save:
                if day not in protocol_df['day'].unique():
                    raise ValueError("The day " + str(day) + " is not in the data")
                
    sampling_frequency = protocol.sampling_frequency
    seconds_per_day = 24*60*60
    data_len = int(seconds_per_day/(1/sampling_frequency))

    columns = (numpy.arange(0, data_len)/sampling_frequency)/3600
    columns = [str(round(column, 2)) for column in columns]
    data_frame = pandas.DataFrame(columns = columns)

    for day in days_to_save:
        day_data = protocol_df[protocol_df['day'] == day]
        index = day_data['real_date'][0]

        if len(day_data) == data_len:
            data_frame.loc[index] = day_data['values'].values
        else:
            print("The data for the day " + str(day) + " is not complete")

    if save_folder != None:
        save_file = save_folder + '/data_per_day_' + protocol_name.lower().replace(' ', '_') + '_' + save_suffix + '.xlsx'
        data_frame.to_excel(save_file)
        
        text = protocol.info_text
        text += "colect_data_per_day parameters:\n"
        text += "days_to_save: " + str(days_to_save) + "\n"
        text += "save_folder: " + str(save_folder) + "\n"
        text += "save_suffix: " + str(save_suffix) + "\n"
        save_text_file = save_folder + '/data_per_day_' + protocol_name.lower().replace(' ', '_') + '_' + save_suffix + '_info.txt'
        with open(save_text_file, 'w') as f:
            f.write(text)
    else:
        return data_frame

def _acrophase_ci_in_zt(best_models):
    """
    Get the acrophase in ZT

    :param acrophase: Acrophase in hours
    :type acrophase: float
    :param period: Period in hours
    :type period: float
    :return: Acrophase in ZT
    :rtype: float
    """
    periods = numpy.array(best_models['period'])
    acrophases = numpy.array(best_models['acrophase'])
    acrophases_ci = list(best_models['CI(acrophase)'])

    acrophases_lower = []
    acrophases_upper = []
    for values in acrophases_ci:
        if isinstance(values, list):
            acrophases_lower.append(values[0])
            acrophases_upper.append(values[1])
        else:
            acrophases_lower.append(numpy.nan)
            acrophases_upper.append(numpy.nan)
    
    lower_diff = numpy.abs(acrophases - acrophases_lower)
    upper_diff = numpy.abs(acrophases_upper - acrophases)

    apply_positive_rad = numpy.vectorize(positive_rad)
    acrophases = apply_positive_rad(acrophases)
    # acrophases_lower = apply_positive_rad(acrophases_lower)
    # acrophases_upper = apply_positive_rad(acrophases_upper)

    acrophases_zt = periods - (acrophases*periods)/(2*numpy.pi)
    # acrophases_lower_zt = periods - (acrophases_lower*periods)/(2*numpy.pi)
    # acrophases_upper_zt = periods - (acrophases_upper*periods)/(2*numpy.pi)

    acrophases_lower_zt = acrophases_zt - (lower_diff*periods)/(2*numpy.pi)
    acrophases_upper_zt = acrophases_zt + (upper_diff*periods)/(2*numpy.pi)

    best_models['acrophase_zt'] = acrophases_zt
    best_models['acrophase_zt_lower'] = acrophases_lower_zt
    best_models['acrophase_zt_upper'] = acrophases_upper_zt

    return best_models

def total_activity_per_day(protocol, save_folder = None, save_suffix = ''):
    print(protocol)
    protocol_df = protocol.data[['values','day']].copy()

    sum_by_group = protocol_df.groupby('day')['values'].sum()

    if save_folder != None:
        save_file = save_folder + '/total_activity_' + protocol.name.lower().replace(' ', '_') + '_' + save_suffix + '.xlsx'
        sum_by_group.to_excel(save_file)
    
        text = protocol.info_text
        text += "total_activity_per_day parameters:\n"
        text += "save_folder: " + str(save_folder) + "\n"
        text += "save_suffix: " + str(save_suffix) + "\n"
        save_text_file = save_folder + '/total_activity_' + protocol.name.lower().replace(' ', '_') + '_' + save_suffix + '_info.txt'
        with open(save_text_file, 'w') as f:
            f.write(text)    
    else:
        save_file = None

def data_series_each_day(protocol):
    each_day_data = {}

    for test_label in protocol.test_labels:
        data = protocol.data.loc[protocol.data['test_labels'] == test_label]
        each_day_data[test_label] = []
        seconds_per_day = 24*60*60
        data_len = seconds_per_day/(1/protocol.sampling_frequency)
        for day in data['day'].unique():
            data_day = data.loc[data['day'] == day]['values']
            if len(data_day) == int(data_len):
                each_day_data[test_label].append(data_day.values)
        
        each_day_data[test_label] = numpy.array(each_day_data[test_label])

    return each_day_data

def fit_cosinor(protocol, dict = None, save_folder = None, save_suffix = ''):
    """
    Fit cosinor model to the data using the CosinorPy library.

    :param protocol: The protocol to fit the cosinor model to, if 0, the average of all protocols is used, defaults
        to 1
    :type protocol: int
    :param dict: A dictionary containing the parameters to fit the cosinor model with keys: record_type, time_shape,
        time_window,
        step, start_time, end_time, n_components. If None, the default values are used, defaults to None
    :type dict: dict
    :param save: If True, the cosinor model is saved in the cosinor_models folder, defaults to True
    :type save: bool
    :return: Dataframe containing the cosinor model parameters
    :rtype: pandas.DataFrame
    """
    warnings.filterwarnings("ignore")
    
    dict_default = {'time_shape': 'continuous',
                    'step': 0.1,
                    'start_time': 20,
                    'end_time': 30,
                    'n_components': [1]}

    if dict != None:
        for key in dict:
            if key in dict_default:
                dict_default[key] = dict[key]
            else:
                raise ValueError("The key " + key + " is not valid")

    time_shape = dict_default['time_shape']
    step = dict_default['step']
    start_time = dict_default['start_time']
    end_time = dict_default['end_time']
    n_components = dict_default['n_components']

    if start_time <= 0:
        raise ValueError("Start time must be greater than 0")

    period = numpy.arange(start_time, end_time, step)

    protocol_df = protocol.get_cosinor_df(time_shape = time_shape)
    protocol_name = protocol.name.replace('_', ' ').capitalize()

    result = cosinor.fit_group(protocol_df, n_components = n_components, period = period, plot = False)
    best = cosinor.get_best_fits(result, n_components = n_components)

    if len(best) == 0:
        best.loc[0] = pandas.np.nan
        best_models_extended = best
    else:
        best_models_extended = cosinor.analyse_best_models(protocol_df, best, analysis = "CI")

    best_models_extended.insert(1, 'first_hour', protocol_df['x'][0])
    best_models_extended = _set_significant_results(best_models_extended)
    best_models_extended = _acrophase_ci_in_zt(best_models_extended) #, protocol_df['y'][0])
    best_models_extended.reset_index(inplace = True, drop = True)

    if save_folder != None:
        save_file = save_folder + '/cosinor_' + protocol_name.lower().replace(' ', '_') + '_' + save_suffix + '.xlsx'
        best_models_extended.to_excel(save_file)
    
        text = protocol.info_text
        text += "fit_cosinor parameters:\n"
        text += "time_shape: " + str(time_shape) + "\n"
        text += "step: " + str(step) + "\n"
        text += "start_time: " + str(start_time) + "\n"
        text += "end_time: " + str(end_time) + "\n"
        text += "n_components: " + str(n_components) + "\n"
        text += "save_folder: " + str(save_folder) + "\n"
        text += "save_suffix: " + str(save_suffix) + "\n"
        save_text_file = save_folder + '/cosinor_' + protocol_name.lower().replace(' ', '_') + '_' + save_suffix + '_info.txt'
        with open(save_text_file, 'w') as f:
            f.write(text)    
    else:
        save_file = None

    warnings.filterwarnings("always")
    
    best_models_extended['test'] = best_models_extended['test'].astype(str)                                             # Change column label type to string
    return best_models_extended, save_file

def fit_cosinor_fixed_period(protocol, best_models, save_folder = None, save_suffix = ''):
    """
    Plot the cosinor period and acrophase for each day of the protocol

    :param best_models_per_day: The best models per day (output of the function get_cosinor_per_day)
    :type best_models_per_day: dict
    """
    warnings.filterwarnings("ignore")

    protocol_df = protocol.get_cosinor_df(time_shape = 'mean')
    protocol_name = protocol.name.replace('_', ' ').capitalize()

    best_models_per_day = pandas.DataFrame()

    change_test_day = protocol.cycle_days
    change_test_day = numpy.cumsum([0] + change_test_day[:-1])

    for count, label in enumerate(best_models['test']):
        test_df = protocol_df[protocol_df['test'] == label]
        index = test_df.index

        period = best_models['period'][count]

        days = []
        for i in range(0, len(index)):
            day = str(index[i].day)
            if len(day) == 1:
                day = '0' + day
            month = str(index[i].month)
            if len(month) == 1:
                month = '0' + month
            year = str(index[i].year)

            days.append(year + '-' + month + '-' + day)

        set_of_days = sorted(list(set(days)))
        test_df['day'] = days

        for count, day in enumerate(range(0, len(set_of_days))):
            day_df = test_df[test_df['day'] == set_of_days[day]]

            result_day = cosinor.fit_group(day_df, n_components = [1], period = [period], plot = False)
            best_model_day = cosinor.get_best_fits(result_day, n_components = [1])

            if len(best_model_day) == 0 or best_model_day['amplitude'][0] <= 0.01:                                      # If the model can't be fitted or the amplitude is too low (threshold), the model isn't considered
                best_model_day.loc[0] = numpy.nan
                best_model_day_extended = best_model_day
            else:
                best_model_day_extended = cosinor.analyse_best_models(day_df, best_model_day, analysis="CI")

            best_model_day_extended.insert(1, 'day', set_of_days[day])
            best_model_day_extended.insert(2, 'first_hour', day_df['x'][0])
            best_models_per_day = pandas.concat([best_models_per_day, best_model_day_extended], axis=0) 

    best_models_per_day = _set_significant_results(best_models_per_day)                                                 # Create a column indicating if the results are significant
    best_models_per_day = _acrophase_ci_in_zt(best_models_per_day)
    best_models_per_day.reset_index(drop = True, inplace = True)                                                        # Reset the index

    if save_folder != None:
        save_file = save_folder + '/cosinor_per_day_fixed_period_' + protocol_name.lower().replace(' ', '_') + '_' + save_suffix + '.xlsx'
        best_models_per_day.to_excel(save_file)
        
        text = protocol.info_text
        text += "fit_cosinor_fixed_period parameters:\n"
        text += "save_folder: " + str(save_folder) + "\n"
        text += "save_suffix: " + str(save_suffix) + "\n"
        save_text_file = save_folder + '/cosinor_per_day_fixed_period_' + protocol_name.lower().replace(' ', '_') + '_' + save_suffix + '_info.txt'
        with open(save_text_file, 'w') as f:
            f.write(text)
    else:
        save_file = None

    warnings.filterwarnings("always")
    best_models_per_day['test'] = best_models_per_day['test'].astype(str)                                                       # Change column label type to string
    return best_models_per_day, save_file

def fit_cosinor_per_day(protocol, dict = None, plot = False, save_folder = None, save_suffix = ''):
    """
    Fits a cosinor model to the data for each day of the protocol

    :param protocol: The protocol to fit the cosinor model parameters for, if 0, the average of all protocols is
        used, defaults to 1
    :type protocol: int
    :param dict: A dictionary containing the parameters to fit the cosinor model with keys: record_type, time_shape,
        time_window,
        step, start_time, end_time, n_components. If None, the default values are used, defaults to None
    :type dict: dict
    """
    warnings.filterwarnings("ignore")

    dict_default = {'day_window': 1,
                    'step': 0.1,
                    'start_time': 20,
                    'end_time': 30,
                    'n_components': [1]}

    if dict != None:
        for key in dict:
            if key in dict_default:
                dict_default[key] = dict[key]
            else:
                raise ValueError("The key " + key + " is not valid")

    day_window = dict_default['day_window']
    step = dict_default['step']
    start_time = dict_default['start_time']
    end_time = dict_default['end_time']
    n_components = dict_default['n_components']

    if start_time <= 0:
        raise ValueError("Start time must be greater than 0")

    periods = numpy.arange(start_time, end_time, step)

    protocol_df = protocol.get_cosinor_df(time_shape = 'mean')
    protocol_name = protocol.name.replace('_', ' ').capitalize()

    best_models_per_day = pandas.DataFrame()                                                                            # Dataframe containing the best model parameters for each day
    protocol_df['test'] = 'all'                                                                                         # Set the test label as 'all' for all the data
    index = protocol_df.index                                                                                           # Get the index (date) of the data

    days = []                                                                                                           # Create a list to store the days
    for i in range(0, len(index)):                                                                                      # Loop through the dates in the index
        day = str(index[i].day)                                                                                         # Get the day
        if len(day) == 1:                                                                                               # If the day is a single digit, add a 0 in front
            day = '0' + day                                                                                             # This is to make sure the date is in the format YYYY-MM-DD
        month = str(index[i].month)                                                                                     # Get the month
        if len(month) == 1:                                                                                             # If the month is a single digit, add a 0 in front
            month = '0' + month                                                                                         # This is to make sure the date is in the format YYYY-MM-DD
        year = str(index[i].year)                                                                                       # Get the year

        days.append(year + '-' + month + '-' + day)                                                                     # Add the date to the list of days

    set_of_days = sorted(list(set(days)))                                                                               # Get the unique days in the data and sort them

    if plot:                                                                                                            # If plot is True, create a figure
        fig, ax = plt.subplots(round(len(set_of_days)/10) + 1, 10, figsize=(40, 40), sharey = True)                     # Create a figure with a subplot for each day
        ax = ax.flatten()                                                                                               # Flatten the axes to make it easier to loop through them

    protocol_df['day'] = days                                                                                           # Add the day column to the dataframe

    for count, day in enumerate(range(0, len(set_of_days) - day_window + 1)):                                           # Loop through the days
        day_df = protocol_df[protocol_df['day'] == set_of_days[day]]                                                    # Get the data for the current day

        if day_window >= 2:                                                                                             # If the user want to use data from multiple days (set by day_window)
            for d in range(1, day_window):                                                                              # Loop through the days
                day_for_window_df = protocol_df[protocol_df['day'] == set_of_days[day + d]]                             # Get the data from the next day
                day_for_window_df['x'] = day_for_window_df['x'] + 24 * d                                                # Add 24 hours to the time for the next day
                day_df = pandas.concat([day_df, day_for_window_df], axis=0)                                             # Concatenate the current day data with the data from the next day

        result_day = cosinor.fit_group(day_df, n_components = n_components, period = periods, plot = False)             # Fit the cosinor model to the data for the current day
        best_model_day = cosinor.get_best_fits(result_day, n_components = n_components)                                 # Get the best model parameters for the current day

        if len(best_model_day) == 0 or best_model_day['amplitude'][0] <= 0.1:                                           #  If the model can't be fitted or the amplitude is too low (threshold), the model isn't considered
            best_model_day.loc[0] = numpy.nan
            best_model_day_extended = best_model_day
            best_model_day_extended['CI(acrophase)'] = numpy.nan
        else:
            best_model_day_extended = cosinor.analyse_best_models(day_df, best_model_day, analysis="CI")        
        
        best_model_day_extended.insert(1, 'day', set_of_days[day])                                                      # Add the day to the best model parameters dataframe
        best_model_day_extended.insert(2, 'first_hour', day_df['x'][0])                                                 # Add the first hour of the day to the best model parameters dataframe   
        
        best_model_day_extended = _acrophase_ci_in_zt(best_model_day_extended)                                          # Reset the index of the dataframe
        best_models_per_day = pandas.concat([best_models_per_day, best_model_day_extended], axis=0)                     # Concatenate the best model parameters for the current day with the best model parameters for all the days

        if plot:                                                                                                        # If plot is True
            ax[count].bar(day_df['x'], day_df['y'], color = 'dimgray')
            ticks_to_plot = numpy.linspace(day_df['x'][0], day_df['x'][-1], num=5, endpoint=True)                       # Create a list of time points change the ticks on the x-axis
            ticks_to_plot = numpy.round(ticks_to_plot, 0)                                                               # Round the time points to 2 decimal places

            m_p_value = best_model_day_extended['p'][0]

            if not numpy.isnan(m_p_value) and m_p_value < 0.05:            
                m_acrophase = best_model_day_extended['acrophase'][0]                                                   # Get the acrophase estimate
                m_period = best_model_day_extended['period'][0]                                                         # Get the period estimate
                m_acrophase_zt = best_model_day_extended['acrophase_zt'][0]                                             # Convert the acrophase to zt

                m_frequency = 1/(m_period)                                                                              # Get the frequency estimate
                m_amplitude = best_model_day_extended['amplitude'][0]                                                   # Get the amplitude estimate
                model = m_amplitude*numpy.cos(numpy.multiply(2*numpy.pi*m_frequency, day_df['x']) + m_acrophase)        # Get the model
                offset = best_model_day_extended['mesor'][0]                                                            # Get the mesor
                model = model + offset                                                                                  # Add the mesor to the model

                ax[count].plot(day_df['x'], model, color = 'midnightblue', linewidth = 3)
                ax[count].axvline(m_acrophase_zt, color = 'black', linestyle = '--', linewidth = 1)
                ax[count].set_title(set_of_days[day] + '\n(PR: ' + str(round(m_period, 2))
                                + ', AC: ' + str(round(m_acrophase_zt, 2)) + ')', fontsize = 20)
            else:
                ax[count].set_title(set_of_days[day] + '\n(PR: NS, AC: NS)', fontsize = 20)
            
            ax[count].set_xticks(ticks_to_plot)
            ax[count].tick_params(axis='both', which='major', labelsize=20)
            ax[count].spines[['right', 'top']].set_visible(False)

    if plot:
        for c in range(len(ax)):
            if c > count:
                ax[c].axis('off')

        fig.suptitle('COSINOR MODEL PER DAY - ' + protocol_name.upper(), fontsize = 40)
        fig.supxlabel('TIME (ZT)', fontsize = 30)
        fig.supylabel('AMPLITUDE', fontsize = 30)
        plt.tight_layout(rect=[0.02, 0.01, 0.98, 0.98])

    if plot and save_folder == None:
        plt.show()

    if save_folder != None:
        save_file = save_folder + '/cosinor_per_day_w' + str(day_window) + '_' + protocol_name.lower().replace(' ', '_') + '_' + save_suffix + '.xlsx'
        best_models_per_day.to_excel(save_file, index = False)
        
        text = protocol.info_text
        text += "fit_cosinor_per_day parameters:\n"
        text += "day_window: " + str(day_window) + "\n"
        text += "step: " + str(step) + "\n"
        text += "start_time: " + str(start_time) + "\n"
        text += "end_time: " + str(end_time) + "\n"
        text += "n_components: " + str(n_components) + "\n"
        text += "save_folder: " + str(save_folder) + "\n"
        text += "save_suffix: " + str(save_suffix) + "\n"
        save_text_file = save_folder + '/cosinor_per_day_w' + str(day_window) + '_' + protocol_name.lower().replace(' ', '_') + '_' + save_suffix + '_info.txt'
        with open(save_text_file, 'w') as f:
            f.write(text)

        if plot:
            plt.savefig(save_folder + '/cosinor_per_day_w' + str(day_window) + '_' + 
                        protocol_name.lower().replace(' ', '_') + '_' + save_suffix + '.png', backend=None)
            plt.close()
    else:
        save_file = None

    warnings.filterwarnings("always")
    best_models_per_day['test'] = best_models_per_day['test'].astype(str)                                               # Change column label type to string
    return best_models_per_day, save_file

def _set_significant_results(best_models_extended):
    """
    Set the significant results of the cosinor analysis

    :param best_models_extended: The extended results of the cosinor analysis
    :type best_models_extended: pandas.DataFrame
    :return: The extended results of the cosinor analysis with the significant results
    :rtype: pandas.DataFrame
    """
    best_models_extended.insert(0, 'significant', 0)

    best_models_extended.loc[(best_models_extended['p'] <= 0.05) & 
                            (best_models_extended['p(amplitude)'] <= 0.05) & 
                            (best_models_extended['p(acrophase)'] <= 0.05) & 
                            (best_models_extended['p(mesor)'] <= 0.05) &
                            (best_models_extended['amplitude'] > 0.01), 'significant'] = 1

    return best_models_extended

def derivate_acrophase(best_models_per_day):
    
    acrophases_zt = numpy.array(best_models_per_day['acrophase_zt'].interpolate(method = 'spline', limit_direction = 'both', order = 3))
    acrophases_zt_smooth = savgol_filter(acrophases_zt, window_length = 10, polyorder = 3, mode = 'nearest')

    first_derivate = numpy.diff(acrophases_zt_smooth)

    plt.plot(first_derivate)
    plt.plot(acrophases_zt)
    plt.plot(acrophases_zt_smooth)

def cbt_cycles(protocol, resample_to = '1H', monving_average_window = 3, std_multiplier = 1, minimal_peak_distance = 10, 
               plot_adjustment_lines = False, save_folder = None, save_suffix = '', format = 'png', labels = ['', 'Time (Hours)', 'Measurement'],
               labels_fontsize = [14, 12, 12], ticks_fontsize = [10, 10]):
    """
    Get the CBT cycles

    :param protocol: The protocol to get the CBT cycles
    :type protocol: protocol
    :param save_folder: The folder to save the data, defaults to None
    :type save_folder: str
    :param save_suffix: The suffix to add to the save file, defaults to ''
    :type save_suffix: str
    """
    if not isinstance(labels, list):
        raise ValueError("labels must be a list.")
    else:
        if len(labels) != 3:
            raise ValueError("labels must be a list with 3 elements (title, x_label, y_label)")
        for label in labels:
            if not isinstance(label, str):
                raise ValueError("labels must be a list of strings")
    if not isinstance(save_suffix, str):
        raise ValueError("save_suffix must be a string.")
    if not isinstance(save_folder, str) and save_folder != None:
        raise ValueError("save_folder must be a string or None.")
    if not isinstance(format, str) and format != 'png' and format != 'svg':
        raise ValueError("format must be 'png' or 'svg'.")
    else:
        format = '.' + format
    if not isinstance(labels_fontsize, list):
        raise ValueError("labels_fontsize must be a list.")
    else:
        if len(labels_fontsize) != 3:
            raise ValueError("labels_fontsize must be a list with 3 elements (title, x_label, y_label)")
        for label in labels_fontsize:
            if not isinstance(label, int):
                raise ValueError("labels_fontsize must be a list of integers")
    if not isinstance(ticks_fontsize, list):
        raise ValueError("ticks_fontsize must be a list.")
    else:
        if len(ticks_fontsize) != 2:
            raise ValueError("ticks_fontsize must be a list with 2 elements (x_ticks, y_ticks)")
        for label in ticks_fontsize:
            if not isinstance(label, int):
                raise ValueError("ticks_fontsize must be a list of integers")

    title = labels[0]
    title_fontsize = labels_fontsize[0]
    x_label = labels[1]
    x_label_fontsize = labels_fontsize[1]
    x_label_ticks = ticks_fontsize[0]
    y_label = labels[2]
    y_label_fontsize = labels_fontsize[2]
    y_label_ticks = ticks_fontsize[1]

    protocol_df = copy.deepcopy(protocol)
    
    sampling_frequency = protocol_df.sampling_frequency
    print(1/sampling_frequency)

    protocol_df.resample(resample_to, method = 'sum')
    protocol_df.apply_filter('moving_average', window = monving_average_window)
    data = protocol_df.data
    
    sampling_frequency = protocol_df.sampling_frequency
    print(1/sampling_frequency)

    protocol_name = protocol_df.name.replace('_', ' ').capitalize()

    test_labels = data['test_labels'].unique()
    
    peak_values = {test_label: [] for test_label in test_labels}
    peak_indexes = {test_label: [] for test_label in test_labels}
    peak_hours = {test_label: [] for test_label in test_labels}
    nadir_values = {test_label: [] for test_label in test_labels}
    nadir_indexes = {test_label: [] for test_label in test_labels}
    nadir_hours = {test_label: [] for test_label in test_labels}

    period_between_nadirs = {test_label: [] for test_label in test_labels}

    columns = 1
    rows = len(test_labels)//columns + len(test_labels)%columns 

    fig, axs = plt.subplots(rows, columns, figsize = (12, 8))
    fig_2, axs_2 = plt.subplots(rows, columns, figsize = (6, 8))

    output = pandas.DataFrame()

    to_hour = 3600/(1/sampling_frequency)

    for count, test_label in enumerate(test_labels):
        data_label = data.loc[data['test_labels'] == test_label]
        
        data_values = data_label['values'].values
        data_hours = numpy.arange(0, len(data_values))/to_hour

        mean_data = numpy.mean(data_values)
        std_data = numpy.std(data_values)

        distance = (minimal_peak_distance*3600)*sampling_frequency

        peaks, _ = find_peaks(data_values, height = std_multiplier*std_data, distance = distance)
        nadirs, _ = find_peaks(- data_values, height = - std_multiplier*std_data, distance = distance)

        peak_values[test_label] = data_values[peaks]
        peak_indexes[test_label] = peaks
        peak_hours[test_label] = data_hours[peaks]

        nadir_values[test_label] = data_values[nadirs]
        nadir_indexes[test_label] = nadirs
        nadir_hours[test_label] = data_hours[nadirs]

        if plot_adjustment_lines:
            axs[count].axhline(mean_data, color = 'black', linestyle = '--', linewidth = 0.5)
            axs[count].axhline(mean_data + std_multiplier*std_data, color = 'black', linestyle = '--', linewidth = 0.5)
            axs[count].axhline(mean_data - std_multiplier*std_data, color = 'black', linestyle = '--', linewidth = 0.5)
            
        axs[count].plot(data_hours, data_values, color = 'dimgray', linewidth = 0.5)
        axs[count].scatter(peak_indexes[test_label]/to_hour, peak_values[test_label], facecolor = 'black', edgecolor = 'black', s = 36, alpha = 1, zorder = 1000,  marker='^')
        axs[count].scatter(nadir_indexes[test_label]/to_hour, nadir_values[test_label], facecolor = 'black', edgecolor = 'black', s = 36, alpha = 1, zorder = 1000,  marker='v')

        axs_c = axs[count].twinx()

        for count_nadir, (start, end) in enumerate(zip(nadir_indexes[test_label][0:-1], nadir_indexes[test_label][1:])):
            period_between_nadirs = (end - start)/to_hour
            
            if count_nadir == 0:
                axs_c.axvline(start/to_hour, color = 'black', linestyle = '--', linewidth = 1, dashes=(5, 10))
            axs_c.axvline(end/to_hour, color = 'black', linestyle = '--', linewidth = 1, dashes=(5, 10))

            cumulative_data = numpy.cumsum(data_values[start:end + 1])
            axs_2[count].plot(numpy.arange(0, end-start)/to_hour, data_values[start:end], color = 'dimgray', linewidth = 0.5)

            max_activity = cumulative_data[-1]

            t25 = numpy.where(cumulative_data >= max_activity*0.25)[0][0]
            t50 = numpy.where(cumulative_data >= max_activity*0.50)[0][0]
            t75 = numpy.where(cumulative_data >= max_activity*0.75)[0][0]

            axs_c.plot(numpy.arange(start, end + 1)/to_hour, cumulative_data, color = 'maroon', linewidth = 2)
            axs_c.scatter((start + t25)/to_hour, cumulative_data[t25], facecolor = 'maroon', edgecolor = 'maroon', s = 16, alpha = 1)
            axs_c.axvline((start + t25)/to_hour, color = 'maroon', linestyle = '--', linewidth = 1)
            axs[count].scatter((start + t25)/to_hour, data_values[start + t25], facecolor = 'midnightblue', edgecolor = 'midnightblue', s = 16, alpha = 1)
            
            axs_c.scatter((start + t50)/to_hour, cumulative_data[t50], facecolor = 'maroon', edgecolor = 'maroon', s = 16, alpha = 1)
            axs_c.axvline((start + t50)/to_hour, color = 'maroon', linestyle = '--', linewidth = 1)
            axs[count].scatter((start + t50)/to_hour, data_values[start + t50], facecolor = 'midnightblue', edgecolor = 'midnightblue', s = 16, alpha = 1)
            
            axs_c.scatter((start + t75)/to_hour, cumulative_data[t75], facecolor = 'maroon', edgecolor = 'maroon', s = 16, alpha = 1)
            axs_c.axvline((start + t75)/to_hour, color = 'maroon', linestyle = '--', linewidth = 1)
            axs[count].scatter((start + t75)/to_hour, data_values[start + t75], facecolor = 'midnightblue', edgecolor = 'midnightblue', s = 16, alpha = 1)

            t25 = t25/to_hour
            t50 = t50/to_hour
            t75 = t75/to_hour

            pandas_dict = {'test_label': test_label, 'interval_inter_nadir': count_nadir, 'interval_start_hr': start/to_hour, 'interval_end_hr': end/to_hour, 'period': period_between_nadirs, 't25': t25, 't50': t50, 't75': t75}
            output = pandas.concat([output, pandas.DataFrame(pandas_dict, index = [0])], axis = 0)

        median_t25 = numpy.median(output.loc[output['test_label'] == test_label]['t25'])
        first_quartile_t25 = numpy.percentile(output.loc[output['test_label'] == test_label]['t25'], 25)
        third_quartile_t25 = numpy.percentile(output.loc[output['test_label'] == test_label]['t25'], 75)

        median_t50 = numpy.median(output.loc[output['test_label'] == test_label]['t50'])
        first_quartile_t50 = numpy.percentile(output.loc[output['test_label'] == test_label]['t50'], 25)
        third_quartile_t50 = numpy.percentile(output.loc[output['test_label'] == test_label]['t50'], 75)
        median_t75 = numpy.median(output.loc[output['test_label'] == test_label]['t75'])
        first_quartile_t75 = numpy.percentile(output.loc[output['test_label'] == test_label]['t75'], 25)
        third_quartile_t75 = numpy.percentile(output.loc[output['test_label'] == test_label]['t75'], 75)
        median_period = numpy.median(output.loc[output['test_label'] == test_label]['period'])
        first_quartile_period = numpy.percentile(output.loc[output['test_label'] == test_label]['period'], 25)
        third_quartile_period = numpy.percentile(output.loc[output['test_label'] == test_label]['period'], 75)

        y_range = axs[count].get_ylim()[1] - axs[count].get_ylim()[0]
        x_range = axs[count].get_xlim()[1] - axs[count].get_xlim()[0]
        y_max = axs[count].get_ylim()[1]

        axs_2[count].axvline(median_t25, color = 'maroon', linestyle = '--', linewidth = 1)
        axs_2[count].text(median_t25 + 0.001*x_range, y_max, 't25', color = 'black', fontsize = x_label_ticks)
        axs_2[count].axvspan(first_quartile_t25, third_quartile_t25, color = 'maroon', alpha = 0.1)
        axs_2[count].axvline(median_t50, color = 'maroon', linestyle = '--', linewidth = 1)
        axs_2[count].text(median_t50 + 0.001*x_range, y_max, 't50', color = 'black', fontsize = x_label_ticks)
        axs_2[count].axvspan(first_quartile_t50, third_quartile_t50, color = 'maroon', alpha = 0.1)
        axs_2[count].axvline(median_t75, color = 'maroon', linestyle = '--', linewidth = 1)
        axs_2[count].text(median_t75 + 0.001*x_range, y_max, 't75', color = 'black', fontsize = x_label_ticks)
        axs_2[count].axvspan(first_quartile_t75, third_quartile_t75, color = 'maroon', alpha = 0.1)
        axs_2[count].axvline(median_period, color = 'midnightblue', linestyle = '--', linewidth = 1)
        axs_2[count].text(median_period + 0.001*x_range, y_max, 'Period', color = 'black', fontsize = x_label_ticks)
        axs_2[count].axvspan(first_quartile_period, third_quartile_period, color = 'midnightblue', alpha = 0.1)

        axs[count].set_title(test_label)
        axs[count].set_ylabel(y_label.upper(), color = 'dimgray')
        axs_c.set_ylabel('CUMULATIVE\n' + y_label.upper(), color = 'maroon')
        axs[-1].set_xlabel('TIME (HOURS)')
        fig.suptitle(title + '\n\n')

        axs[count].title.set_size(title_fontsize)
        axs[count].xaxis.label.set_size(x_label_fontsize)
        axs[count].yaxis.label.set_size(y_label_fontsize)
        axs_c.yaxis.label.set_size(y_label_fontsize)
        axs[count].tick_params(axis='x', labelsize = x_label_ticks)
        axs[count].tick_params(axis='y', labelsize = y_label_ticks)
        axs_c.tick_params(axis='y', labelsize = y_label_ticks)

        legend_elements = [Line2D([0], [0], marker = '^', color = 'black', markerfacecolor = 'black', markersize = 8, label = 'Peaks'),
                           Line2D([0], [0], marker = 'v', color = 'black', markerfacecolor = 'black', markersize = 8, label = 'Nadirs'),
                           Line2D([0], [0], linestyle = '--', color = 'black', label = 'Period between nadirs'),
                           Line2D([0], [0], linestyle = '--', color = 'maroon', label = 't25, t50, t75'),
                           Line2D([0], [0], linestyle = '-', color = 'maroon', label = 'Cumulative measurement', linewidth = 2),
                           Line2D([0], [0], marker = 'o', color='midnightblue', markerfacecolor = 'midnightblue', markersize = 6, label = 't25, t50, t75 intersection')]
        
        fig.legend(handles = legend_elements, loc = 'upper right', bbox_to_anchor = (0.99, 1), fontsize = 12, frameon = False, ncol = 2)

        axs_2[count].set_title(test_label)
        axs_2[count].set_ylabel(y_label.upper(), color = 'dimgray', fontsize = 10)
        axs_2[-1].set_xlabel(x_label.upper())
        axs_2[count].set_ylim(axs[count].get_ylim()[0] - 0.1*y_range, axs[count].get_ylim()[1] + 0.1*y_range)
        
        axs_2[count].title.set_size(title_fontsize)
        axs_2[count].xaxis.label.set_size(x_label_fontsize)
        axs_2[count].yaxis.label.set_size(y_label_fontsize)
        axs_2[count].tick_params(axis='x', labelsize = x_label_ticks)
        axs_2[count].tick_params(axis='y', labelsize = y_label_ticks)

        fig.tight_layout()
        fig_2.tight_layout()

    if save_folder != None:
        if format != '.both':
            fig.savefig(save_folder + '/cbt_cycles_' + protocol_name.lower().replace(' ', '_') + '_' + save_suffix + format, backend=None)
            fig_2.savefig(save_folder + '/cbt_cycles_intervals_' + protocol_name.lower().replace(' ', '_') + '_' + save_suffix + format, backend=None)
        else:
            fig.savefig(save_folder + '/cbt_cycles_' + protocol_name.lower().replace(' ', '_') + '_' + save_suffix + '.png', backend=None)
            fig_2.savefig(save_folder + '/cbt_cycles_intervals_' + protocol_name.lower().replace(' ', '_') + '_' + save_suffix + '.png', backend=None)
            fig.savefig(save_folder + '/cbt_cycles_' + protocol_name.lower().replace(' ', '_') + '_' + save_suffix + '.svg', backend=None)
            fig_2.savefig(save_folder + '/cbt_cycles_intervals_' + protocol_name.lower().replace(' ', '_') + '_' + save_suffix + '.svg', backend=None)
            
        plt.close(fig)
        plt.close(fig_2)

        save_file = save_folder + '/cbt_cycles_' + protocol_name.lower().replace(' ', '_') + '_' + save_suffix + '.xlsx'
        output.to_excel(save_file, index = False)
        
        text = protocol.info_text
        text += "cbt_cycles parameters:\n"
        text += "save_folder: " + str(save_folder) + "\n"
        text += "save_suffix: " + str(save_suffix) + "\n"
        save_text_file = save_folder + '/cbt_cycles_' + protocol_name.lower().replace(' ', '_') + '_' + save_suffix + '_info.txt'
        with open(save_text_file, 'w') as f:
            f.write(text)
    else:
        plt.show()


# file_activity = "F:\\github\\chronobiology_analysis\\protocols\\data_expto_5\\data\\1_control_dl\\control_dl_ale_animal_01.asc"
# file_temperature = "F:\\github\\chronobiology_analysis\\protocols\\data_expto_5\\data\\1_control_dl\\control_dl_temp_animal_01.asc"

# protocol = protocol('test', file_activity, file_temperature, 20, 18, 'DL', 'control')
# protocol.get_cosinor_df(time_shape = 'median', time_window = 24)
# pass