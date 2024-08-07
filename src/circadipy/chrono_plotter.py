import numpy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import pandas
import math
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, periodogram, welch

plt.ion()

# def get_group_average(self):
#     """
#     Get the average of the protocols
#     :return: Dataframe with the average of the protocols
#     :rtype: pandas.DataFrame
#     """
#     protocol_mean = pandas.DataFrame()
#     for protocol in self.protocols:
#         protocol_mean = pandas.concat((protocol_mean, protocol.data))
#     protocol_mean = protocol_mean.groupby(protocol_mean.index).mean()

#     return protocol_mean

def time_serie(protocol, labels = ['Time Series', 'Time (Days)', 'Amplitude'],
               color = 'midnightblue', save_folder = None, save_suffix = '', format = 'png',
               labels_fontsize = [14, 12, 12], ticks_fontsize = [10, 10]):
    '''
    Plot the time series of the protocol

    :param protocol: Protocol object to be vizualized
    :type protocol: read_protocol
    :param labels: List with the labels of the plot [title, x_label, y_label], defaults to ['Time Series', 'Time (Days)', 'Amplitude']
    :type labels: list
    :param color: Color of the plot, defaults to 'midnightblue'
    :type color: str
    :param save_folder: Path to save the folder if it is None, the plot is just shown, defaults to None
    :type save_folder: str
    :save_suffix: Suffix to add to the file name, defaults to ''
    :type save_suffix: str
    :param format: Format to save the figure (png or svg), defaults to 'png'
    :type format: str
    :param labels_fontsize: List with the fontsize of the labels [title, x_label, y_label], defaults to [14, 12, 12]
    :type labels_fontsize: list
    :param ticks_fontsize: List with the fontsize of the ticks [x_ticks, y_ticks], defaults to [12, 10, 10]
    :type ticks_fontsize: list
    '''
    if not isinstance(labels, list):
        raise ValueError("labels must be a list.")
    else:
        if len(labels) != 3:
            raise ValueError("labels must be a list with 3 elements (title, x_label, y_label)")
        for label in labels:
            if not isinstance(label, str):
                raise ValueError("labels must be a list of strings")
    if not isinstance(color, str):
        raise ValueError("color must be a string")
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

    protocol_name = protocol.name.replace('_', ' ').capitalize()

    fig = plt.figure(figsize = (12, 5))                                                                                 # Create a figure
    ax = fig.add_subplot(111)                                                                                           # Add a subplot
    ax.plot(protocol.data['values'], color = color, linewidth = 2)                                                      # Plot the data with noise
    ax.set_xlabel(x_label)                                                                                            # Set the x label
    ax.set_ylabel(y_label)                                                                                              # Set the y label
    ax.set_title(title)                                                                                                 # Set the title

    ax.title.set_size(title_fontsize)
    ax.xaxis.label.set_size(x_label_fontsize)
    ax.yaxis.label.set_size(y_label_fontsize)
    ax.tick_params(axis='x', labelsize = x_label_ticks)
    ax.tick_params(axis='y', labelsize = y_label_ticks)

    if save_folder == None:
        plt.show()
    else:
        if format != '.both':
            plt.savefig(save_folder + '/time_serie_' + protocol_name.lower() + '_' + save_suffix + format, backend=None)
            plt.close()
        else:
            plt.savefig(save_folder + '/time_serie_' + protocol_name.lower() + '_' + save_suffix + '.svg', backend=None)
            plt.savefig(save_folder + '/time_serie_' + protocol_name.lower() + '_' + save_suffix + '.png', backend=None)
            plt.close()
        
        text = protocol.info_text
        text += "time_serie parameters:\n"
        text += "title: " + title + "\n"
        text += "x_label: " + x_label + "\n"
        text += "y_label: " + y_label + "\n"
        text += "color: " + color + "\n"
        text += "save_folder: " + str(save_folder) + "\n"
        text += "save_suffix: " + save_suffix + "\n"
        text += "format: " + format + "\n"
        save_text_folder = save_folder + '/time_serie_' + protocol_name.lower() + '_' + save_suffix + '_info.txt'
        with open(save_text_folder, 'w') as file:
            file.write(text)

def mean_time_series(protocol, labels = ['Mean Time Series', 'Samples', 'Amplitude']):
    '''
    Plot the mean time series of the protocol

    :param protocol: Protocol object to be vizualized
    :type protocol: read_protocol
    :param labels: List with the labels of the plot [title, x_label, y_label], defaults to ['Mean Time Series', 'Time (Days)', 'Amplitude']
    :type labels: list
    '''
    if not isinstance(labels, list):
        raise ValueError("labels must be a list.")
    else:
        if len(labels) != 3:
            raise ValueError("labels must be a list with 3 elements (title, x_label, y_label)")
        for label in labels:
            if not isinstance(label, str):
                raise ValueError("labels must be a list of strings")

    title = labels[0]
    x_label = labels[1]
    y_label = labels[2]

    protocol_name = protocol.name.replace('_', ' ').capitalize()

    fig = plt.figure(figsize = (12, 5))                                                                                 # Create a figure
    ax = fig.add_subplot(111)                                                                                           # Add a subplot

    for test_label in protocol.test_labels:
        data = protocol.data.loc[protocol.data['test_labels'] == test_label]
        data_sum = []
        seconds_per_day = 24*60*60
        data_len = seconds_per_day/(1/protocol.sampling_frequency)
        for day in data['day'].unique():
            data_day = data.loc[data['day'] == day]['values']
            if len(data_day) == int(data_len):
                data_sum.append(data_day)

        data_mean = numpy.mean(data_sum, axis = 0)
        data_std = numpy.std(data_sum, axis = 0)

        ax.plot(data_mean, label = test_label)                                                                          # Plot the data with noise
        ax.fill_between(range(len(data_mean)), data_mean - data_std, data_mean + data_std, alpha = 0.3)

    ax.set_xlabel(x_label)                                                                                              # Set the x label
    ax.set_ylabel(y_label)                                                                                              # Set the y label
    ax.set_title(title)                                                                                                 # Set the title
    ax.legend()                                                                                                         # Add the legend
    fig.suptitle(protocol_name)                                                                                         # Set the figure title

    plt.show()

def time_serie_sum_per_day(protocol, labels = ['Sum of Time Series Per Day', 'Time (Days)', 'Amplitude'], 
                           color = 'midnightblue', save_folder = None, save_suffix = '', format = 'png',
                           labels_fontsize = [14, 12, 12], ticks_fontsize = [10, 10]):
    '''
    Plot the time series of the protocol

    :param protocol: Protocol object to be vizualized
    :type protocol: read_protocol
    :param labels: List with the labels of the plot [title, x_label, y_label], defaults to ['Sum of Time Series Per Day', 'Time (Days)', 'Amplitude']
    :type labels: list
    :param color: Color of the plot, defaults to 'midnightblue'
    :type color: str
    :param save_folder: Path to save the folder if it is None, the plot is just shown, defaults to None
    :type save_folder: str
    :save_suffix: Suffix to add to the file name, defaults to ''
    :type save_suffix: str
    :param format: Format to save the figure (png or svg), defaults to 'png'
    :type format: str
    :param labels_fontsize: List with the fontsize of the labels [title, x_label, y_label], defaults to [14, 12, 12]
    :type labels_fontsize: list
    :param ticks_fontsize: List with the fontsize of the ticks [x_ticks, y_ticks], defaults to [12, 10, 10]
    :type ticks_fontsize: list
    '''
    if not isinstance(labels, list):
        raise ValueError("labels must be a list.")
    else:
        if len(labels) != 3:
            raise ValueError("labels must be a list with 3 elements (title, x_label, y_label)")
        for label in labels:
            if not isinstance(label, str):
                raise ValueError("labels must be a list of strings")
    if not isinstance(color, str):
        raise ValueError("color must be a string")
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

    protocol_name = protocol.name.replace('_', ' ').capitalize()

    sumation = protocol.data['values'].groupby(protocol.data.index.date).sum()
    min_value = 0
    max_value = numpy.nanmax(sumation) + 0.1*numpy.nanmax(sumation)

    fig = plt.figure(figsize = (12, 5))                                                                                 # Create a figure
    ax = fig.add_subplot(111)                                                                                           # Add a subplot
    ax.plot(sumation, color = color, linewidth = 2)                                                                     # Plot the data with noise
    ax.set_xlabel(x_label)                                                                                              # Set the x label
    ax.set_ylabel(y_label)                                                                                              # Set the y label
    ax.set_title(title)                                                                                                 # Set the title
    ax.set_ylim([min_value, max_value])                                                                                 # Set the y limits

    ax.title.set_size(title_fontsize)
    ax.xaxis.label.set_size(x_label_fontsize)
    ax.yaxis.label.set_size(y_label_fontsize)
    ax.tick_params(axis='x', labelsize = x_label_ticks)
    ax.tick_params(axis='y', labelsize = y_label_ticks)

    if save_folder == None:
        plt.show()
    else:
        if format != '.both':
            plt.savefig(save_folder + '/time_serie_sum_per_day_' + protocol_name.lower() + '_' + save_suffix + format,
                        backend=None)
            plt.close()
        else:
            plt.savefig(save_folder + '/time_serie_sum_per_day_' + protocol_name.lower() + '_' + save_suffix + '.svg',
                    backend=None)
            plt.savefig(save_folder + '/time_serie_sum_per_day_' + protocol_name.lower() + '_' + save_suffix + '.png',
                    backend=None)
            plt.close()
        
        text = protocol.info_text
        text += "time_serie_sum_per_day parameters:\n"
        text += "title: " + title + "\n"
        text += "x_label: " + x_label + "\n"
        text += "y_label: " + y_label + "\n"
        text += "color: " + color + "\n"
        text += "save_folder: " + str(save_folder) + "\n"
        text += "save_suffix: " + save_suffix + "\n"
        text += "format: " + format + "\n"
        save_text_folder = save_folder + '/time_serie_sum_per_day_' + protocol_name.lower() + '_' + save_suffix + '_info.txt'
        with open(save_text_folder, 'w') as file:
            file.write(text)

def actogram_bar(protocol, first_hour = 0, save_folder = None, save_suffix = '',
                 adjust_figure = [1, 0.95, 0.85, 0.2, 0.05], norm_value = None, format = 'png',
                 x_label = 'HOURS', labels_fontsize = [14, 12, 12], ticks_fontsize = [8, 8]):
    """
    Plot the actogram of the protocol in a bar plot

    :param protocol: Protocol object to be vizualized
    :type protocol: read_protocol
    :param first_hour: First hour of the day to be plotted, if different from 0, the actogram is shifted, defaults to 0
    :type first_hour: int
    :param save_folder: Path to save the folder if it is None, the plot is just shown, defaults to None
    :type save_folder: str
    :save_suffix: Suffix to add to the file name, defaults to ''
    :type save_suffix: str
    :param adjust_figure: List with the parameters to adjust the figure [column_height, rigth, top, bottom, left],
        defaults to [1, 0.95, 0.85, 0.2, 0.05]
    :type adjust_figure: list
    :param norm_value: List with the minimum and maximum values to be used in to y axis limits (it is used when the data have nan valuesa and the data variabilty is too high), defaults to None
    :param format: Format to save the figure (png or svg), defaults to 'png'
    :type format: str
    :param x_label: Label of the x axis, defaults to 'HOURS'
    :type x_label: str
    :param labels_fontsize: List with the fontsize of the labels [title, x_label, y_label], defaults to [14, 12, 12]
    :type labels_fontsize: list
    :param ticks_fontsize: List with the fontsize of the ticks [x_ticks, y_ticks], defaults to [12, 10, 10]
    :type ticks_fontsize: list
    """
    if not isinstance(first_hour, int) and first_hour >= 24 and first_hour < 0:
        raise ValueError("First hour must be an integer less than 24 and greater than 0.")
    if not isinstance(adjust_figure, list) and len(adjust_figure) != 5:
        raise ValueError("adjust_figure must be a list with 5 values.")
    if not isinstance(save_suffix, str):
        raise ValueError("save_suffix must be a string.")
    if not isinstance(save_folder, str) and save_folder != None:
        raise ValueError("save_folder must be a string or None.")
    if not isinstance(norm_value, list) and norm_value != None:
        raise ValueError("norm_value must be None or a list with two values (min and max) in which min < max")
    if isinstance(norm_value, list):
        if len(norm_value) != 2 or norm_value[0] > norm_value[1]:
            raise ValueError("norm_value must be None or a list with two values (min and max) in which min < max")
    if not isinstance(format, str) and format != 'png' and format != 'svg':
        raise ValueError("format must be 'png' or 'svg'.")
    else:
        format = '.' + format
    if not isinstance(x_label, str):
        raise ValueError("x_label must be a string")
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

    adjust_height = adjust_figure[0]
    adjust_right = adjust_figure[1]
    adjust_top = adjust_figure[2]
    adjust_bottom = adjust_figure[3]
    adjust_left = adjust_figure[4]

    title_fontsize = labels_fontsize[0]
    x_label_fontsize = labels_fontsize[1]
    x_ticks_fontsize = ticks_fontsize[0]
    y_label_fontsize = labels_fontsize[2]
    y_ticks_fontsize = ticks_fontsize[1]

    actogram = protocol.data.copy()
    protocol_name = protocol.name.replace('_', ' ').capitalize()

    # Change 0 to nan
    # actogram['values'] = numpy.where(actogram['values'] == numpy.nan, 0, actogram['values'])

    displacement_before = numpy.where(actogram.index.hour == first_hour)[0][0]

    first_hour_correction = numpy.timedelta64(24 - first_hour, 'h')

    actogram.index = [date + first_hour_correction for date in actogram.index]
    actogram['day'] = [str(row).split(" ")[0] for row in actogram.index]

    displacement = numpy.where(actogram.index.hour == first_hour)[0][0]
    displacement = numpy.abs(displacement_before - displacement)

    days = list(actogram['day'].unique())
    num_days = len(days)
    actogram['day_steps'] = actogram.index.hour
    actogram['day_steps'] = actogram['day_steps'] + actogram.index.minute/60
    step_length = actogram['day_steps'][1] - actogram['day_steps'][0]

    if norm_value == None:
        min_value = numpy.nanmin(actogram['values'])
        max_value = numpy.nanmax(actogram['values']) 
    elif len(norm_value) == 2 and norm_value[0] < norm_value[1] and norm_value[0] != None and norm_value[1] != None:
        min_value = norm_value[0]
        max_value = norm_value[1]
    else:
        raise ValueError("norm_value must be None or a list with two values (min and max)")

    total_height = 0.1*num_days*adjust_height
    total_width = 7
    fig, subplots = plt.subplots(num_days, 2, squeeze=False, sharex=True, sharey=True,
                                    figsize=(total_width + 3, total_height),
                                    gridspec_kw={'hspace': 0, 'wspace': 0})

    hours_per_tick = 2
    tick_pos = list(range(0, 24, hours_per_tick))

    tick_labels = tick_pos[int(first_hour/hours_per_tick):] + tick_pos[:int(first_hour/hours_per_tick)]
    tick_labels = [str(hour) for hour in tick_labels]

    for count, day in enumerate(days):
        axes = []
        if count > 0 and count < num_days:
            axes.append(subplots[count][0])
            axes.append(subplots[count-1][1])
        elif count == 0:
            axes.append(subplots[count][0])
        else:
            axes.append(subplots[count-1][1])

        time_to_plot = actogram[actogram['day'] == day]['day_steps']
        activity_to_plot = actogram[actogram['day'] == day]['values']
        is_night_to_plot = actogram[actogram['day'] == day]['is_night']

        for count_ax, ax in enumerate(axes):
            for step, time in zip(is_night_to_plot, time_to_plot):
                if step == True:
                    ax.axvspan(time, time + step_length, facecolor='lightgray', alpha=0.9, edgecolor='none')
                else:
                    ax.axvspan(time, time + step_length, facecolor='white', alpha=1, edgecolor='none')

            ax.bar(time_to_plot, activity_to_plot, width = step_length, align='edge', color='black',
                    edgecolor='none')
            ax.set_ylim([min_value, max_value])
            if count_ax != 0 and count%2 != 0:
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_ylabel("Day " + str(count + 1), rotation=0, fontsize=y_ticks_fontsize, labelpad=20, va='center')
            elif count_ax == 0 and count%2 == 0:
                ax.yaxis.set_label_position("left")
                ax.yaxis.tick_left()
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_ylabel("Day " + str(count + 1), rotation=0, fontsize=y_ticks_fontsize, labelpad=20, va='center')

    ax.set_xlim([0, 24])
    #ax.set_ylim([min_value, max_value + 0.1*max_value])
    subplots[-1][0].set_xticks(tick_pos, tick_labels, fontsize=x_ticks_fontsize)
    subplots[-1][1].set_xticks(tick_pos, tick_labels, fontsize=x_ticks_fontsize)

    fig.subplots_adjust(right=adjust_right, top=adjust_top, bottom=adjust_bottom, left=adjust_left)

    fig.suptitle('ACTOGRAM - ' + protocol_name.upper(), y = 0.95, fontsize=title_fontsize)
    fig.supxlabel(x_label, y = 0.05, fontsize=x_label_fontsize)

    if save_folder == None:
        plt.show()
    else:
        if format != '.both':
            plt.savefig(save_folder + '/actogram_bar_' + protocol_name.lower() + '_' + save_suffix + format, backend=None)
            plt.close()
        else:
            plt.savefig(save_folder + '/actogram_bar_' + protocol_name.lower() + '_' + save_suffix + '.svg', backend=None)
            plt.savefig(save_folder + '/actogram_bar_' + protocol_name.lower() + '_' + save_suffix + '.png', backend=None)
            plt.close()

        text = protocol.info_text
        text += "actogram_bar parameters:\n"
        text += "first_hour: " + str(first_hour) + "\n"
        text += "save_folder: " + str(save_folder) + "\n"
        text += "save_suffix: " + save_suffix + "\n"
        text += "adjust_figure: " + str(adjust_figure) + "\n"
        text += "norm_value: " + str(norm_value) + "\n"
        text += "format: " + format + "\n"
        save_text_folder = save_folder + '/actogram_bar_' + protocol_name.lower() + '_' + save_suffix + '_info.txt'
        with open(save_text_folder, 'w') as file:
            file.write(text)

def actogram_colormap(protocol, first_hour = 0, unit_of_measurement = "Amplitude", save_folder = None, save_suffix = '',
                      adjust_figure = [1, 0.95, 0.85, 0.2, 0.05], norm_color = None, format = 'png', x_label = 'HOURS',
                      labels_fontsize = [14, 12, 12], ticks_fontsize = [8, 8]):
    """
    Plot the actogram of the protocol in a colormap

    :param protocol: Protocol object to be vizualized
    :type protocol: read_protocol
    :param first_hour: First hour of the day to be plotted, if different from 0, the actogram is shifted, defaults to 0
    :type first_hour: int
    :param unit_of_measurement: Unit of measurement of the protocol, defaults to "Amplitude". This is used to label the
        colorbar
    :type unit_of_measurement: str
    :param save_folder: Path to save the folder if it is None, the plot is just shown, defaults to None
    :type save_folder: str
    :save_suffix: Suffix to add to the file name, defaults to ''
    :type save_suffix: str
    :param adjust_figure: List with the parameters to adjust the figure [column_height, rigth, top, bottom, left],
        defaults to [1, 0.95, 0.85, 0.2, 0.05]
    :type adjust_figure: list
    :param norm_color: List with the minimum and maximum values to be used in the colormap, defaults to [33, 40]
    :type norm_color: list
    :param format: Format to save the figure (png or svg), defaults to 'png'
    :type format: str
    :param x_label: Label of the x axis, defaults to 'HOURS'
    :type x_label: str
    :param labels_fontsize: List with the fontsize of the labels [title, x_label, y_label], defaults to [14, 12, 12]
    :type labels_fontsize: list
    :param ticks_fontsize: List with the fontsize of the ticks [x_ticks, y_ticks], defaults to [12, 10, 10]
    :type ticks_fontsize: list
    """
    if not isinstance(first_hour, int) and first_hour >= 24 and first_hour < 0:
        raise ValueError("First hour must be an integer less than 24 and greater than 0.")
    if not isinstance(adjust_figure, list) and len(adjust_figure) != 5:
        raise ValueError("adjust_figure must be a list with 5 values.")
    if not isinstance(save_suffix, str):
        raise ValueError("save_suffix must be a string.")
    if not isinstance(save_folder, str) and save_folder != None:
        raise ValueError("save_folder must be a string or None.")
    if not isinstance(norm_color, list) and norm_color != None:
        raise ValueError("norm_color must be None or a list with two values (min and max) in which min < max")
    if isinstance(norm_color, list):
        if len(norm_color) != 2 or norm_color[0] > norm_color[1]:
            raise ValueError("norm_color must be None or a list with two values (min and max) in which min < max")
    if not isinstance(format, str) and format != 'png' and format != 'svg':
        raise ValueError("format must be 'png' or 'svg'.")
    else:
        format = '.' + format
    if not isinstance(x_label, str):
        raise ValueError("x_label must be a string")
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

    adjust_height = adjust_figure[0]
    adjust_right = adjust_figure[1]
    adjust_top = adjust_figure[2]
    adjust_bottom = adjust_figure[3]
    adjust_left = adjust_figure[4]

    title_fontsize = labels_fontsize[0]
    x_label_fontsize = labels_fontsize[1]
    x_ticks_fontsize = ticks_fontsize[0]
    y_label_fontsize = labels_fontsize[2]
    y_ticks_fontsize = ticks_fontsize[1]

    actogram = protocol.data.copy()
    protocol_name = protocol.name.replace('_', ' ').capitalize()

    # Change 0 to nan
    # actogram['values'] = numpy.where(actogram['values'] < 30, numpy.nan, actogram['values'])

    displacement_before = numpy.where(actogram.index.hour == first_hour)[0][0]

    first_hour_correction = numpy.timedelta64(24 - first_hour, 'h')
    actogram.index = [date + first_hour_correction for date in actogram.index]
    actogram['day'] = [str(row).split(" ")[0] for row in actogram.index]

    displacement = numpy.where(actogram.index.hour == first_hour)[0][0]
    displacement = numpy.abs(displacement_before - displacement)

    days = list(actogram['day'].unique())
    num_days = len(days)
    actogram['day_steps'] = actogram.index.hour
    actogram['day_steps'] = actogram['day_steps'] + actogram.index.minute/60
    step_length = actogram['day_steps'][1] - actogram['day_steps'][0]

    max_value = numpy.nanmax(actogram['values'])
    min_value = numpy.nanmin(actogram['values'])

    if norm_color == None:
        norm = mpl.colors.Normalize(vmin = min_value, vmax = max_value)
    elif len(norm_color) == 2 and norm_color[0] < norm_color[1] and norm_color[0] != None and norm_color[1] != None:
        norm = mpl.colors.Normalize(vmin = norm_color[0], vmax = norm_color[1])
    else:
        raise ValueError("norm_color must be None or a list with two values (min and max)")
    cmap = cm.YlOrRd
    m = cm.ScalarMappable(norm = norm, cmap = cmap)

    total_height = 0.1*num_days*adjust_height
    total_width = 8
    fig, subplots = plt.subplots(num_days, 2, squeeze=False, sharex=True, sharey=True,
                                    figsize=(total_width + 3, total_height),
                                    gridspec_kw={'hspace': 0, 'wspace': 0})

    hours_per_tick = 2
    tick_pos = list(range(0, 24, hours_per_tick))
    tick_labels = tick_pos[int(first_hour/hours_per_tick):] + tick_pos[:int(first_hour/hours_per_tick)]
    tick_labels = [str(hour) for hour in tick_labels]

    for count, day in enumerate(days):
        axes = []
        if count > 0 and count < num_days:
            axes.append(subplots[count][0])
            axes.append(subplots[count-1][1])
        elif count == 0:
            axes.append(subplots[count][0])
        else:
            axes.append(subplots[count-1][1])

        time_to_plot = actogram[actogram['day'] == day]['day_steps']
        value_to_plot = actogram[actogram['day'] == day]['values']
        is_night_to_plot = actogram[actogram['day'] == day]['is_night']

        for count_ax, ax in enumerate(axes):
            for value, time, night in zip(value_to_plot, time_to_plot, is_night_to_plot):
                color = m.to_rgba(value)
                if numpy.isnan(value):
                    color = 'darkgray'
                # if not night:
                #     ax.scatter(time, 0, color='white', s=15, edgecolors='none')
                # else:
                #     ax.scatter(time, 0, color='darkgray', s=15, edgecolors='none')
                ax.axvspan(time, time + step_length, facecolor=color, alpha=1, edgecolor='none')
            ax.set_ylim([0, max_value])
            if count_ax != 0 and count%2 != 0:
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.set_ylabel("Day " + str(count + 1), rotation=0, fontsize=y_ticks_fontsize, labelpad=20, va='center')
            elif count_ax == 0 and count%2 == 0:
                ax.yaxis.set_label_position("left")
                ax.yaxis.tick_left()
                ax.set_ylabel("Day " + str(count + 1), rotation=0, fontsize=y_ticks_fontsize, labelpad=20, va='center')

    for count, ax in enumerate(subplots.flatten()):
        if count != 0 and count != 1:
            ax.spines['top'].set_visible(False)
        if count != len(subplots.flatten()) - 1 and count != len(subplots.flatten()) - 2:
            ax.spines['bottom'].set_visible(False)
        ax.set_yticks([])
        ax.set_xticks([])

    ax.set_xlim([0, 24])
    #ax.set_ylim([0, max_value])
    subplots[-1][0].set_xticks(tick_pos, tick_labels, fontsize=x_ticks_fontsize)
    subplots[-1][1].set_xticks(tick_pos, tick_labels, fontsize=x_ticks_fontsize)

    fig.subplots_adjust(right=adjust_right, top=adjust_top, bottom=adjust_bottom, left=adjust_left)

    fig.suptitle('ACTOGRAM - ' + protocol_name.upper(), y = 0.95, x = 0.4, fontsize=title_fontsize)
    fig.supxlabel(x_label, y = 0.05, x = 0.4, fontsize=x_label_fontsize)
    cax, kw = mpl.colorbar.make_axes([a for a in subplots.flat], orientation='vertical',
                                       anchor=(0.95, 0.5), aspect=25)
    cax.tick_params(labelsize=y_label_fontsize)
    cbar = plt.colorbar(m, cax=cax, **kw)
    cbar.set_label(unit_of_measurement.upper(), rotation=270, labelpad=20, fontsize=10)

    if save_folder == None:
        plt.show()
    else:
        if format != '.both':
            plt.savefig(save_folder+'/actogram_cmap_'+ protocol_name.lower() + '_' + save_suffix + format, backend=None)
            plt.close()
        else:
            plt.savefig(save_folder+'/actogram_cmap_'+ protocol_name.lower() + '_' + save_suffix + '.svg', backend=None)
            plt.savefig(save_folder+'/actogram_cmap_'+ protocol_name.lower() + '_' + save_suffix + '.png', backend=None)
            plt.close()

        text = protocol.info_text
        text += "actogram_colormap parameters:\n"
        text += "first_hour: " + str(first_hour) + "\n"
        text += "unit_of_measurement: " + unit_of_measurement + "\n"
        text += "save_folder: " + str(save_folder) + "\n"
        text += "save_suffix: " + save_suffix + "\n"
        text += "adjust_figure: " + str(adjust_figure) + "\n"
        text += "norm_color: " + str(norm_color) + "\n"
        text += "format: " + format + "\n"
        save_text_folder = save_folder + '/actogram_cmap_' + protocol_name.lower() + '_' + save_suffix + '_info.txt'
        with open(save_text_folder, 'w') as file:
            file.write(text)

    return fig

def data_periodogram(protocol, time_shape = 'continuous', method = 'periodogram', max_period = 48,
                     unit_of_measurement = 'Unit', save_folder = None, save_suffix = '', format = 'png', 
                     x_label = 'PERIOD (HOUR)', labels_fontsize = [14, 12, 12], ticks_fontsize = [10, 10]):
    """
    Plot the periodogram of the protocol

    :param protocol: Protocol object to be vizualized
    :type protocol: read_protocol
    :param time_shape: Time shape to be used, can be 'continuous', 'median' or 'mean', defaults to 'continuous'. If
        'continuous', the periodogram is calculated using the whole protocol data, if 'median' or 'mean', the periodogram
        is calculates using the median or mean of the data per day
    :type time_shape: str
    :param method: Method to be used to calculate the periodogram, can be 'periodogram' or 'welch', defaults to
        'periodogram'
    :type method: str
    :param max_period: Maximum period to be plotted, defaults to 48
    :type max_period: int
    :param unit_of_measurement: Unit of measurement of the protocol, defaults to "Unit". This is used to label the
        y axis
    :type unit_of_measurement: str
    :param save_folder: Path to save the folder if it is None, the plot is just shown, defaults to None
    :type save_folder: str
    :save_suffix: Suffix to add to the file name, defaults to ''
    :type save_suffix: str
    :param format: Format to save the figure (png or svg), defaults to 'png'
    :type format: str
    :param x_label: Label of the x axis, defaults to 'PERIOD (HOUR)'
    :type x_label: str
    :param labels_fontsize: List with the fontsize of the labels [title, x_label, y_label], defaults to [14, 12, 12]
    :type labels_fontsize: list
    :param ticks_fontsize: List with the fontsize of the ticks [x_ticks, y_ticks], defaults to [10, 10]
    :type ticks_fontsize: list
    
    """
    if not isinstance(time_shape, str) and time_shape != 'continuous' and time_shape != 'mean' and time_shape != 'median':
        raise ValueError("Time shape must be 'continuous', 'median' or 'mean'")
    if not isinstance(method, str) and method != 'periodogram' and method != 'welch':
        raise ValueError("Method must be 'periodogram' or 'welch'")
    if not isinstance(max_period, int) and max_period < 0:
        raise ValueError("max_period must be a positive integer")
    if not isinstance(unit_of_measurement, str):
        raise ValueError("unit_of_measurement must be a string")
    if not isinstance(save_suffix, str):
        raise ValueError("save_suffix must be a string.")
    if not isinstance(save_folder, str) and save_folder != None:
        raise ValueError("save_folder must be a string or None.")
    if not isinstance(format, str) and format != 'png' and format != 'svg':
        raise ValueError("format must be 'png' or 'svg'.")
    else:
        format = '.' + format
    if not isinstance(x_label, str):
        raise ValueError("x_label must be a string")
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

    title_fontsize = labels_fontsize[0]
    x_label_fontsize = labels_fontsize[1]
    x_ticks_fontsize = ticks_fontsize[0]
    y_label_fontsize = labels_fontsize[2]
    y_ticks_fontsize = ticks_fontsize[1]

    _test = protocol.data['test_labels']
    test_labels, indexes = numpy.unique(_test, return_index = True)
    _test_labels = [x for _ ,x in sorted(zip(indexes, test_labels))]

    if len(_test_labels) == 1:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7), sharey=True, sharex=True)
        axes = [ax]
    else:
        fig, ax = plt.subplots(len(_test_labels)//2 + len(_test_labels)%2, 2, figsize=(7, 7), sharey=True, sharex=True)
        axes = ax.flatten()

    protocol_df = protocol.get_cosinor_df(time_shape = time_shape)
    protocol_name = protocol.name.replace('_', ' ').capitalize()

    save_data = pandas.DataFrame()

    for count, test_label in enumerate(_test_labels):
        record = protocol_df[protocol_df['test'] == test_label]['y']
        time_in_hour = protocol_df[protocol_df['test'] == test_label]['x']
        sampling_interval_hour = time_in_hour[1] - time_in_hour[0]

        if time_shape != 'continuous':
            time_to_mean = numpy.unique(time_in_hour)
            mean_record = []
            for time in time_to_mean:
                if time_shape == 'median':
                    mean_record.append(numpy.median(record[time == time_in_hour]))
                elif time_shape == 'mean':
                    mean_record.append(numpy.mean(record[time == time_in_hour]))
            record = mean_record

        if method == 'periodogram':
            freq, psd_raw = periodogram(record, 1/sampling_interval_hour)
        elif method == 'welch':
            freq, psd_raw = welch(record, 1/sampling_interval_hour, nperseg=len(record))

        p_value = 0.05
        length = len(record)

        threshold = (1 - (p_value/length)**(1/(length-1))) * sum(psd_raw)

        if freq[0] == 0:
            period = 1/freq[1:]
            psd = psd_raw[1:]
        else:
            period = 1/freq
            psd = psd_raw

        psd = psd[period <= max_period]
        period = period[period <= max_period]
        data = pandas.DataFrame({'test':test_label, 'period': period, 'psd': psd})
        save_data = pandas.concat([save_data, data], axis = 0, ignore_index = True)

        peak_index, _  = find_peaks(psd, height = threshold)

        axes[count].plot([min(period), max(period)], [threshold, threshold], '--', linewidth=2, color = 'dimgray',
                            label = 'THRESHOLD')
        axes[count].plot(period, psd, '-', linewidth = 2, color = 'midnightblue')
        #axes[count].legend(['THRESHOLD'], loc='upper right', fontsize=8)
        axes[count].spines[['right', 'top']].set_visible(False)

        if any(peak_index):
            peak_y = psd[peak_index]
            peak_x = period[peak_index]
            max_peak_y = max(peak_y)
            max_peak_x = peak_x[peak_y == max_peak_y]
            # axes[count].text(max_peak_x, max_peak_y, "{:.2f}   ".format(max_peak_x[0]), fontsize=8,
            #                     verticalalignment='center', horizontalalignment='right')
            axes[count].set_title(test_label.replace('_', ' ').upper() + '\n(PEAK ON ' + "{:.2f}".format(max_peak_x[0]) +
                                ' HOURS)', fontsize=x_label_fontsize)   
        else:
            axes[count].set_title(test_label.replace('_', ' ').upper() + '\n(NO SIGNIFICANT PEAK)', fontsize=8)
        axes[count].title.set_size(title_fontsize)
        axes[count].xaxis.label.set_size(x_label_fontsize)
        axes[count].yaxis.label.set_size(y_label_fontsize)
        axes[count].tick_params(axis='x', labelsize = x_ticks_fontsize)
        axes[count].tick_params(axis='y', labelsize = y_ticks_fontsize)

    fig.supylabel('PSD $(' + unit_of_measurement + '^{2}/HZ)$', fontsize = y_label_fontsize + 2)                                                         # If ativity, can be "COUNTS", if temperature, can be "C°"
    fig.supxlabel(x_label, fontsize = x_label_fontsize + 2)

    if count%2 == 0 and count != 0:
        axes[-1].axis('off')

    plt.suptitle('PERIODOGRAM OF ' + time_shape.upper() + ' - ' + protocol_name.upper(), fontsize=title_fontsize + 2)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.tight_layout()

    if save_folder == None:
        plt.show(block=False)
    else:
        save_data.to_csv(save_folder + '/periodogram_' + protocol_name.lower().replace(' ', '_') + '_' + save_suffix
                         + '.csv', index=False)
        if format != '.both':
            plt.savefig(save_folder + '/periodogram_' + protocol_name.lower().replace(' ', '_') + '_' + save_suffix +
                        format, backend=None)
            plt.close()
        else:
            plt.savefig(save_folder + '/periodogram_' + protocol_name.lower().replace(' ', '_') + '_' + save_suffix +
                    '.svg', backend=None)
            plt.savefig(save_folder + '/periodogram_' + protocol_name.lower().replace(' ', '_') + '_' + save_suffix +
                    '.png', backend=None)
            plt.close()

        text = protocol.info_text
        text += "data_periodogram parameters:\n"
        text += "time_shape: " + time_shape + "\n"
        text += "method: " + method + "\n"
        text += "max_period: " + str(max_period) + "\n"
        text += "unit_of_measurement: " + unit_of_measurement + "\n"
        text += "save_folder: " + str(save_folder) + "\n"
        text += "save_suffix: " + save_suffix + "\n"
        text += "format: " + format + "\n"
        save_text_folder = save_folder + '/periodogram_' + protocol_name.lower().replace(' ', '_') + '_' + save_suffix + '_info.txt'
        with open(save_text_folder, 'w') as file:
            file.write(text)

def model_overview_detailed(protocol, best_models_fixed, only_significant = False, save_folder = None, save_suffix = '',
                            format = 'png', labels_fontsize = [14, 12, 12], ticks_fontsize = [10, 10]):
    '''
    Plot the cosinor period and acrophase for each day of the protocol. This function is used to plot the results of the
    fit_cosinor_fixed_period function. In this case, acrophase (and the others parameters) are calculated fixing the
    period calculated to each stage.

    :param protocol: Protocol object to be vizualized
    :type protocol: read_protocol
    :param best_models_fixed: Cosinor model parameters per day (output of the fit_cosinor_fixed_period function)
    :type best_models_fixed: pandas.DataFrame
    :param only_significant: If True, only the significant days are plotted, defaults to False
    :type only_significant: bool
    :param save_folder: Path to save the folder if it is None, the plot is just shown, defaults to None
    :type save_folder: str
    :save_suffix: Suffix to add to the file name, defaults to ''
    :type save_suffix: str
    :param format: Format to save the figure (png or svg), defaults to 'png'
    :type format: str
    :param labels_fontsize: List with the fontsize of the labels [title, x_label, y_label], defaults to [14, 12, 12]
    :type labels_fontsize: list
    :param ticks_fontsize: List with the fontsize of the ticks [x_ticks, y_ticks], defaults to [10, 10]
    :type ticks_fontsize: list
    '''
    if not isinstance(best_models_fixed, pandas.DataFrame):
        raise ValueError("best_models_fixed must be a pandas.DataFrame")
    if not isinstance(save_suffix, str):
        raise ValueError("save_suffix must be a string.")
    if not isinstance(only_significant, bool):
        raise ValueError("only_significant must be a bool.")
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

    title_fontsize = labels_fontsize[0]
    x_label_fontsize = labels_fontsize[1]
    x_ticks_fontsize = ticks_fontsize[0]
    y_label_fontsize = labels_fontsize[2]
    y_ticks_fontsize = ticks_fontsize[1]

    protocol_name = protocol.name.replace('_', ' ').capitalize()
    test_labels = protocol.test_labels

    change_test_day = protocol.cycle_days
    change_test_day = numpy.cumsum([0] + change_test_day[:-1])

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    days = best_models_fixed['day']
    last_day = None
    count = -1
    days_plot = []
    ticks_plot = []
    for d in days:
        if d != last_day:
            count += 1
            days_plot.append(count)
            ticks_plot.append(d)
            last_day = d
        else:
            days_plot.append(count)
            last_day = d

    days_plot = numpy.array(days_plot) + 1
    change_test_day = numpy.array(change_test_day) + 1

    significance = best_models_fixed['significant']
    if only_significant:
        where_non_significant = numpy.where(significance == 0)[0]
        if where_non_significant.size == 0:
            raise Warning("There are no non-significant days")
    else:
        where_non_significant = numpy.where(significance == 42)[0]

    m_periods = numpy.array(best_models_fixed['period'])
    m_periods[where_non_significant] = numpy.nan

    m_period_min = numpy.nanmin(m_periods)
    m_period_max = numpy.nanmax(m_periods)
    lower_lim = _get_next_previus_odd(m_period_min)[1]
    upper_lim = _get_next_previus_odd(m_period_max)[0]

    ax.plot(days_plot, m_periods, '-o', color='dimgray', markersize = 2)
    ax.set_xlabel('DAYS', fontsize=x_label_fontsize)
    ax.set_ylabel('PERIOD (HOUR)', color = 'dimgray', fontsize=y_label_fontsize)
    ax.set_ylim(lower_lim - 0.1*(upper_lim - lower_lim), upper_lim + 0.1*(upper_lim - lower_lim))
    ax.set_yticks(numpy.arange(lower_lim, upper_lim + 1, 1))
    ax.tick_params(axis='y', labelsize = y_ticks_fontsize)
    ax2 = ax.twinx()

    m_acrophases_zt = numpy.array(best_models_fixed['acrophase_zt'])
    m_acrophases_zt[where_non_significant] = numpy.nan
    m_acrophases_zt_ci_upper = numpy.array(best_models_fixed['acrophase_zt_upper'])
    m_acrophases_zt_ci_upper[where_non_significant] = numpy.nan
    m_acrophases_zt_ci_lower = numpy.array(best_models_fixed['acrophase_zt_lower'])
    m_acrophases_zt_ci_lower[where_non_significant] = numpy.nan

    ax2.plot(days_plot, m_acrophases_zt, '-o', color='midnightblue', markersize = 2)

    ax2.set_ylabel('ACROPHASE (HOUR)', color='midnightblue', fontsize=y_label_fontsize)
    ax2.set_ylim([-12, 36])
    ax2.set_yticks([-6, 0, 6, 12, 18, 24, 30])
    ax2.set_yticklabels(['18', '00', '06', '12', '18', '00', '06'])
    ax2.tick_params(axis='y', labelsize = y_ticks_fontsize)

    for x in sorted(set(days_plot))[0::5]:
        ax.axvline(x = x, color='black', linestyle='--', linewidth=0.5, alpha=0.2)
    for l, x in enumerate(change_test_day):
        ax.axvline(x = x, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.annotate(str(l + 1), xy=(x, upper_lim), xytext=(x + 0.2, upper_lim), fontsize=12,
                    color='black', rotation=0)
    for d in range(0, len(days_plot)):
        ax2.plot([days_plot[d], days_plot[d]], [m_acrophases_zt_ci_upper[d], m_acrophases_zt_ci_lower[d]],
                    color='midnightblue', linewidth=0.5, alpha=1)

    ax.set_title('COSINOR MODEL PARAMETERS - ' + protocol_name.upper(), fontsize=title_fontsize)
    ax.tick_params(axis='x', labelsize = x_ticks_fontsize)
    plt.xticks(sorted(set(days_plot))[0::5])
    plt.tight_layout()

    if save_folder == None:
        plt.show(block=False)
    else:
        if format != '.both':
            plt.savefig(save_folder + '/model_parameters_' + protocol_name.lower().replace(' ', '_') + '_' + save_suffix +
                        format, backend=None)
            plt.close()
        else:
            plt.savefig(save_folder + '/model_parameters_' + protocol_name.lower().replace(' ', '_') + '_' + save_suffix +
                        '.svg', backend=None)
            plt.savefig(save_folder + '/model_parameters_' + protocol_name.lower().replace(' ', '_') + '_' + save_suffix +
                        '.png', backend=None)
            plt.close()
        
        text = protocol.info_text
        text += "model_overview_detailed parameters:\n"
        text += "only_significant: " + str(only_significant) + "\n"
        text += "save_folder: " + str(save_folder) + "\n"
        text += "save_suffix: " + save_suffix + "\n"
        text += "format: " + format + "\n"
        save_text_folder = save_folder + '/model_parameters_' + protocol_name.lower().replace(' ', '_') + '_' + save_suffix + '_info.txt'
        with open(save_text_folder, 'w') as file:
            file.write(text)

def model_over_signal(protocol, best_models, position = 'head', mv_avg_window = 1, save_folder = None, save_suffix = '',
                      format = 'png', y_label = 'AMPLITUDE', labels_fontsize = [10, 8, 8], ticks_fontsize = [6, 6]):
    """
    Plot the cosinor model and the protocol data to each stage. Can be used to compare the cosinor model to the signal.

    :param protocol: Protocol object to be vizualized
    :type protocol: read_protocol
    :param best_models: Cosinor model parameters (output of the fit_cosinor function)
    :type best_models: pandas.DataFrame
    :param position: Position of the data to be plotted, can be 'head' or 'tail', defaults to 'head'. If 'tail', the
        last part of the data is plotted inverted (first point is the last point of the data, the second point is the
        penultimate point of the data, etc.)
    :type position: str
    :param mv_avg_window: Window size to apply a moving average to the data, defaults to 1. If 1, no moving average is
        applied.
    :type mv_avg_window: int
    :param save_folder: Path to save the folder if it is None, the plot is just shown, defaults to None
    :type save_folder: str
    :save_suffix: Suffix to add to the file name, defaults to ''
    :type save_suffix: str
    :param format: Format to save the figure (png or svg), defaults to 'png'
    :type format: str
    :param y_label: Label of the y axis, defaults to 'AMPLITUDE'
    :type y_label: str
    :param labels_fontsize: List with the fontsize of the labels [title, x_label, y_label], defaults to [14, 12, 12]
    :type labels_fontsize: list
    :param ticks_fontsize: List with the fontsize of the ticks [x_ticks, y_ticks], defaults to [10, 10]
    :type ticks_fontsize: list
    """
    if not isinstance(best_models, pandas.DataFrame):
        raise ValueError("best_models must be a pandas.DataFrame")
    if not isinstance(position, str) and position != 'head' and position != 'tail':
        raise ValueError("position must be 'head' or 'tail'")
    if not isinstance(mv_avg_window, int) and mv_avg_window < 1:
        raise ValueError("mv_avg_window must be a positive integer (if 1, no moving average is applied)")
    if not isinstance(save_suffix, str):
        raise ValueError("save_suffix must be a string.")
    if not isinstance(save_folder, str) and save_folder != None:
        raise ValueError("save_folder must be a string or None.")
    if not isinstance(format, str) and format != 'png' and format != 'svg':
        raise ValueError("format must be 'png' or 'svg'.")
    else:
        format = '.' + format
    if not isinstance(y_label, str):
        raise ValueError("y_label must be a string")
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

    title_fontsize = labels_fontsize[0]
    x_label_fontsize = labels_fontsize[1]
    x_ticks_fontsize = ticks_fontsize[0]
    y_label_fontsize = labels_fontsize[2]
    y_ticks_fontsize = ticks_fontsize[1]

    protocol_df = protocol.get_cosinor_df(time_shape = 'continuous')
    protocol_name = protocol.name.replace('_', ' ').capitalize()
    best_models['test'] = best_models['test'].astype(str)

    cm = 1/2.54                                                                                                         # Centimeters to inches convertion
    if len(best_models) == 1:                                                                                           # If there is only one test label (e.g. control)
        rows = 1                                                                                                        # Number of rows for the plot
        fig, ax = plt.subplots(rows, 1, figsize=(15*cm, 5*cm), sharey=True)                                             # Create the plot
        ax = [ax]
    else:
        rows = len(best_models)//2 + len(best_models)%2                                                                 # Number of rows for the plot
        fig, ax = plt.subplots(rows, 2, figsize=(2*15*cm, rows*5*cm), sharey=True)                                      # Create the plot
        ax = ax.flatten()                                                                                               # Flatten the array of axes
    
    min_time = float('inf')                                                                                             # Initialize variable to find the minimum time
    min_index = 0                                                                                                       # Initialize the minimum index to 0
    for label in best_models['test']:                                                                                   # For each test label (e.g. control)
        new_min = protocol_df[protocol_df['test'] == label]['x'][-1]                                                    # Get the last time point of the cosinor model
        if new_min < min_time:                                                                                          # If the last time point is smaller than the current minimum
            min_time = new_min                                                                                          # Update the minimum time
            min_index = len(protocol_df[protocol_df['test'] == label]['x'])                                             # Update the minimum index

    min_value = numpy.inf                                                                                               # Initialize variable to find the minimum value
    max_value = -numpy.inf                                                                                              # Initialize variable to find the maximum value

    for label in range(0, len(best_models)):                                                                            # For each test label (e.g. control)
        original_data = protocol_df[protocol_df['test'] == best_models['test'][label]]['y'].values                      # Get the original data (e.g. activity or temperature)
        time = protocol_df[protocol_df['test'] == best_models['test'][label]]['x'].values                               # Get the time of the original data

        if position == 'head':                                                                                          # If the parameter position is 'head'
            original_data = original_data[0:min_index]                                                                  # Get the first part of the original data
            time_to_plot = time[0:min_index]                                                                            # Get the first part of the time
            time = time[0:min_index]                                                                                    # Get the first part of the time
        elif position == 'tail':                                                                                        # If the parameter position is 'tail'
            original_data = numpy.flip(original_data)[0:min_index]                                                      # Get the last part of the original data
            time_to_plot = time[0:min_index]                                                                            # Get the last part of the time
            time = numpy.flip(time)[0:min_index]                                                                        # Get the last part of the time
        else:
            raise ValueError("The position " + position + " is not valid")                                              # If the parameter position is not valid, raise an error

        min_value = min(min_value, min(original_data))                                                                  # Update the minimum value
        max_value = max(max_value, max(original_data))                                                                  # Update the maximum value

        if mv_avg_window > 1:                                                                                           # If the moving average window is larger than 1
            original_data = uniform_filter1d(original_data, size = mv_avg_window)                                       # Apply a moving average to the original data with the given window size

        ax[label].bar(time_to_plot, original_data, color='dimgray')

        significance = best_models['significant'][label]
        m_acrophase = best_models['acrophase'][label]
        m_period = best_models['period'][label]
        m_acrophase_zt = best_models['acrophase_zt'][label]                                                         # Convert the acrophase to the zt scale

        m_frequency = 1/(m_period)

        m_amplitude = best_models['amplitude'][label]
        model = m_amplitude*numpy.cos(numpy.multiply(2*numpy.pi*m_frequency, time) + m_acrophase)

        offset = best_models['mesor'][label]
        model = model + offset

        ax[label].plot(time_to_plot, model, color='midnightblue', linewidth = 3)
        ax[label].axvline(x = m_acrophase_zt, color='black', linestyle='--', linewidth=2, alpha=0.8)
        if significance == 1:                                                                                           # If the p-value is smaller than 0.05
            ax[label].set_title(str(best_models['test'][label]).replace('_', ' ').upper() + '\n(AC: ' +
                                str(round(m_acrophase_zt, 2)) + ', PR: ' + str(round(m_period, 2)) + ')')
        else:
            ax[label].set_title(str(best_models['test'][label]).replace('_', ' ').upper() + '\n(AC: ' +
                                str(round(m_acrophase_zt, 2)) + ', PR: ' + str(round(m_period, 2)) + ' - NS)')
        ax[label].spines[['right', 'top']].set_visible(False)
        ax[label].set_xlim([time_to_plot[0], time_to_plot[-1]])
        ax[label].title.set_size(title_fontsize)
        ax[label].xaxis.label.set_size(x_label_fontsize)
        ax[label].yaxis.label.set_size(y_label_fontsize)
        ax[label].tick_params(axis='x', labelsize = x_ticks_fontsize)
        ax[label].tick_params(axis='y', labelsize = y_ticks_fontsize)


    ax[label].set_ylim(min_value - 0.1*(max_value - min_value), max_value + 0.1*(max_value - min_value))

    if label < len(ax) - 1:
        ax[label + 1].set_visible(False)

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,2))
    fig.suptitle('COSINOR MODEL - ' + protocol_name.upper(), fontsize=title_fontsize + 2)
    fig.supxlabel('TIME (HOURS FROM START OF PROTOCOL)', fontsize=x_label_fontsize + 2)
    fig.supylabel(y_label, fontsize=y_label_fontsize + 2)
    plt.tight_layout()

    if save_folder == None:
        plt.show(block=False)
    else:
        if format != '.both':
            plt.savefig(save_folder + '/model_over_signal_' + position + '_' + protocol_name.lower().replace(' ', '_') +
                        '_' + save_suffix + format, backend=None)
            plt.close()
        else:
            plt.savefig(save_folder + '/model_over_signal_' + position + '_' + protocol_name.lower().replace(' ', '_') +
                    '_' + save_suffix + '.svg', backend=None)
            plt.savefig(save_folder + '/model_over_signal_' + position + '_' + protocol_name.lower().replace(' ', '_') +
                    '_' + save_suffix + '.png', backend=None)
            plt.close()

        text = protocol.info_text
        text += "model_over_signal parameters:\n"
        text += "position: " + position + "\n"
        text += "mv_avg_window: " + str(mv_avg_window) + "\n"
        text += "save_folder: " + str(save_folder) + "\n"
        text += "save_suffix: " + save_suffix + "\n"
        text += "format: " + format + "\n"
        save_text_folder = save_folder + '/model_over_signal_' + position + '_' + protocol_name.lower().replace(' ', '_') + '_' + save_suffix + '_info.txt'
        with open(save_text_folder, 'w') as file:
            file.write(text)

def model_overview(protocol, best_models, only_significant = False, save_folder = None, save_suffix = '', format = 'png',
                   labels_fontsize = [14, 12, 12], ticks_fontsize = [10, 10]):
    """
    Plots the cosinor model pariod and acrophase for each protocol satage. This function is used to plot the results of
    the fit_cosinor function.

    :param protocol: Protocol object to be vizualized
    :type protocol: read_protocol
    :param best_models: The cosinor model parameters (output of the fit_cosinor function)
    :type best_models: pandas.DataFrame
    :param only_significant: If True, only the significant days are plotted, defaults to False
    :type only_significant: bool
    :param save_folder: Path to save the folder if it is None, the plot is just shown, defaults to None
    :type save_folder: str
    :save_suffix: Suffix to add to the file name, defaults to ''
    :type save_suffix: str
    :param format: Format to save the figure (png or svg), defaults to 'png'
    :type format: str
    :param labels_fontsize: List with the fontsize of the labels [title, x_label, y_label], defaults to [14, 12, 12]
    :type labels_fontsize: list
    :param ticks_fontsize: List with the fontsize of the ticks [x_ticks, y_ticks], defaults to [10, 10]
    :type ticks_fontsize: list
    """
    if not isinstance(best_models, pandas.DataFrame):
        raise ValueError("best_models must be a pandas.DataFrame")
    if not isinstance(only_significant, bool):
        raise ValueError("only_significant must be a bool.")
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

    title_fontsize = labels_fontsize[0]
    x_label_fontsize = labels_fontsize[1]
    x_ticks_fontsize = ticks_fontsize[0]
    y_label_fontsize = labels_fontsize[2]
    y_ticks_fontsize = ticks_fontsize[1]

    protocol_name = protocol.name.replace('_', ' ').capitalize()

    significance = best_models['significant']
    if only_significant:
        where_non_significant = numpy.where(significance == 0)[0]
        if where_non_significant.size == 0:
            raise Warning("There are no non-significant days")
    else:
        where_non_significant = numpy.where(significance == 42)[0]

    m_periods = numpy.array(best_models['period'])
    m_periods[where_non_significant] = numpy.nan

    m_period_min = numpy.nanmin(m_periods)
    m_period_max = numpy.nanmax(m_periods)
    lower_lim = _get_next_previus_odd(m_period_min)[1]
    upper_lim = _get_next_previus_odd(m_period_max)[0]

    m_acrophases_zt = numpy.array(best_models['acrophase_zt'])
    m_acrophases_zt[where_non_significant] = numpy.nan

    x_labels = [str(t).replace('_', ' ').upper() for t in best_models['test']]
    x_ticks = range(len(x_labels))

    fig, ax = plt.subplots()
    plt.plot(x_ticks, m_periods, '-o', color = 'dimgray', linewidth=3)
    plt.xlim(-0.5, len(x_ticks) - 0.5)
    plt.xticks(x_ticks, x_labels, rotation=45, ha='right')

    ax.set_title('COSINOR MODEL PARAMETERS - ' + protocol_name.upper() + '\n', fontsize = title_fontsize)
    ax.set_xlabel('PROTOCOL LABELS', fontsize = x_label_fontsize)
    ax.set_ylabel('PERIOD (HOUR)', color = 'dimgray', fontsize = y_label_fontsize)
    ax.set_ylim(lower_lim - 0.1*(upper_lim - lower_lim), upper_lim + 0.1*(upper_lim - lower_lim))
    ax.set_yticks(numpy.arange(lower_lim, upper_lim + 1, 1))
    ax.tick_params(axis='y', labelsize = y_ticks_fontsize)
    ax.tick_params(axis='x', labelsize = x_ticks_fontsize)
    ax2=ax.twinx()

    ax2.plot(x_ticks, m_acrophases_zt, '--o', color = 'midnightblue', linewidth=3)
    ax2.set_ylabel('ACROPHASE (HOUR)', color = 'midnightblue', fontsize = y_label_fontsize)
    ax2.set_ylim(0, 24)
    ax2.set_yticks([0, 6, 12, 18, 24])
    ax2.tick_params(axis='y', labelsize = y_ticks_fontsize)
    ax2.tick_params(axis='x', labelsize = x_ticks_fontsize)
    plt.tight_layout()

    if save_folder == None:
        plt.show(block=False)
    else:
        if format != '.both':
            plt.savefig(save_folder + '/model_overview_' + protocol_name.lower().replace(' ', '_') + '_' + save_suffix +
                        format, backend=None)
            plt.close()
        else:
            plt.savefig(save_folder + '/model_overview_' + protocol_name.lower().replace(' ', '_') + '_' + save_suffix +
                    '.svg', backend=None)
            plt.savefig(save_folder + '/model_overview_' + protocol_name.lower().replace(' ', '_') + '_' + save_suffix +
                    '.png', backend=None)
            plt.close()
        
        text = protocol.info_text
        text += "model_overview parameters:\n"
        text += "only_significant: " + str(only_significant) + "\n"
        text += "save_folder: " + str(save_folder) + "\n"
        text += "save_suffix: " + save_suffix + "\n"
        text += "format: " + format + "\n"
        save_text_folder = save_folder + '/model_overview_' + protocol_name.lower().replace(' ', '_') + '_' + save_suffix + '_info.txt'
        with open(save_text_folder, 'w') as file:
            file.write(text)

def model_per_day(protocol, best_models_per_day, day_window, only_significant = False, save_folder = None, save_suffix = '', format = 'png'):
    """
    Plot the cosinor model parameters per day. This function is used to plot the results of the fit_cosinor_per_day
    function. In this case, parameters are calculated every day (with a moving window or not).

    :param protocol: Protocol object to be vizualized
    :type protocol: read_protocol
    :param best_models_per_day: Cosinor model parameters per day (output of the fit_cosinor_fixed_period function)
    :type best_models_per_day: pandas.DataFrame
    :param day_window: Window size to apply a moving average to the data, defaults to 1. If 1, no moving average is
        applied.
    :type day_window: int
    :param only_significant: If True, only the significant days are plotted, defaults to False
    :type only_significant: bool
    :param save_folder: Path to save the folder if it is None, the plot is just shown, defaults to None
    :type save_folder: str
    :save_suffix: Suffix to add to the file name, defaults to ''
    :type save_suffix: str
    :param format: Format to save the figure (png or svg), defaults to 'png'
    :type format: str
    """
    if not isinstance(best_models_per_day, pandas.DataFrame):
        raise ValueError("best_models_per_day must be a pandas.DataFrame")
    if not isinstance(day_window, int) and day_window < 1:
        raise ValueError("day_window must be a positive integer (if 1, no moving average is applied)")
    if not isinstance(only_significant, bool):
        raise ValueError("only_significant must be a bool.")
    if not isinstance(save_suffix, str):
        raise ValueError("save_suffix must be a string.")
    if not isinstance(save_folder, str) and save_folder != None:
        raise ValueError("save_folder must be a string or None.")
    if not isinstance(format, str) and format != 'png' and format != 'svg':
        raise ValueError("format must be 'png' or 'svg'.")
    else:
        format = '.' + format

    protocol_name = protocol.name.replace('_', ' ').capitalize()

    change_test_day = protocol.cycle_days
    change_test_day = numpy.cumsum([0] + change_test_day[:-1])
    test_labels = protocol.test_labels

    fig, ax = plt.subplots(1, 1, figsize=(10,5))

    days = best_models_per_day['day']
    last_day = None
    count = -1
    days_plot = []
    ticks_plot = []
    for d in days:
        if d != last_day:
            count += 1
            days_plot.append(count)
            ticks_plot.append(d)
            last_day = d
        else:
            days_plot.append(count)
            last_day = d

    days_plot = numpy.array(days_plot) + 1
    change_test_day = numpy.array(change_test_day) + 1

    m_p_value = numpy.array(best_models_per_day['p'])
    if only_significant:
        where_non_significant = numpy.where(m_p_value > 0.05)[0]
        if where_non_significant.size == 0:
            raise Warning("There are no non-significant days")
    else:
        where_non_significant = numpy.where(m_p_value >= 10000)[0]

    m_periods = numpy.array(best_models_per_day['period'])
    m_periods[where_non_significant] = numpy.nan

    m_period_min = numpy.nanmin(m_periods)
    m_period_max = numpy.nanmax(m_periods)
    lower_lim = _get_next_previus_odd(m_period_min)[1]
    upper_lim = _get_next_previus_odd(m_period_max)[0]

    ax.plot(days_plot, m_periods, '-o', color='dimgray', markersize = 2)
    ax.set_xlabel('DAYS', fontsize=12)
    ax.set_ylabel('PERIOD (HOUR)', color = 'dimgray', fontsize=12)
    ax.set_ylim(lower_lim - 0.1*(upper_lim - lower_lim), upper_lim + 0.1*(upper_lim - lower_lim))
    ax.set_yticks(numpy.arange(lower_lim, upper_lim + 1, 1))
    ax2 = ax.twinx()

    m_acrophases_zt = numpy.array(best_models_per_day['acrophase_zt'])
    m_acrophases_zt[where_non_significant] = numpy.nan
    m_acrophases_zt_ci_upper = numpy.array(best_models_per_day['acrophase_zt_upper'])
    m_acrophases_zt_ci_upper[where_non_significant] = numpy.nan
    m_acrophases_zt_ci_lower = numpy.array(best_models_per_day['acrophase_zt_lower'])
    m_acrophases_zt_ci_lower[where_non_significant] = numpy.nan

    ax2.plot(days_plot, m_acrophases_zt, '-o', color='midnightblue', markersize = 2)
    ax2.set_ylabel('ACROPHASE (HOUR)', color='midnightblue', fontsize=12)
    ax2.set_ylim([-12, 36])
    ax2.set_yticks([-12, -6, 0, 6, 12, 18, 24, 30, 36])
    ax2.set_yticklabels(['12', '18', '00', '06', '12', '18', '00', '06', '12'])

    for x in sorted(set(days_plot))[0::5]:
        ax.axvline(x = x, color='black', linestyle='--', linewidth=0.5, alpha=0.2)
    for l, x in enumerate(change_test_day):
        ax.axvline(x = x, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.annotate(str(l + 1), xy=(x, upper_lim), xytext=(x + 0.2, upper_lim), fontsize=12,
                    color='black', rotation=0)
    for d in range(0, len(days_plot)):
        ax2.plot([days_plot[d], days_plot[d]], [m_acrophases_zt_ci_upper[d], m_acrophases_zt_ci_lower[d]],
                    color='midnightblue', linewidth=0.5, alpha=1)

    ax.set_title('COSINOR MODEL PARAMETERS FOR EACH DAY - ' + protocol_name.upper())
    plt.xticks(sorted(set(days_plot))[0::5])
    plt.tight_layout()

    if save_folder == None:
        plt.show(block=False)
    else:
        if format != '.both':
            plt.savefig(save_folder + '/model_parameters_per_day_w' + str(day_window) + '_' +
                        protocol_name.lower().replace(' ', '_') + '_' + save_suffix + format, backend=None)
            plt.close()
        else:
            plt.savefig(save_folder + '/model_parameters_per_day_w' + str(day_window) + '_' +
                        protocol_name.lower().replace(' ', '_') + '_' + save_suffix + '.svg', backend=None)
            plt.savefig(save_folder + '/model_parameters_per_day_w' + str(day_window) + '_' +
                        protocol_name.lower().replace(' ', '_') + '_' + save_suffix + '.png', backend=None)
            plt.close()

        text = protocol.info_text
        text += "model_per_day parameters:\n"
        text += "day_window: " + str(day_window) + "\n"
        text += "only_significant: " + str(only_significant) + "\n"
        text += "save_folder: " + str(save_folder) + "\n"
        text += "save_suffix: " + save_suffix + "\n"
        text += "format: " + format + "\n"
        save_text_folder = save_folder + '/model_parameters_per_day_w' + str(day_window) + '_' +\
                           protocol_name.lower().replace(' ', '_') + '_' + save_suffix + '_info.txt'
        with open(save_text_folder, 'w') as file:
            file.write(text)

def _get_next_previus_odd(number):
    """
    Get the next odd number of the input number

    :param number: The number
    :type number: int
    :return: The next odd number
    :rtype: int
    """
    next_int = math.ceil(number)
    previus_int = math.floor(number)

    if next_int % 2 == 0:
        next_int += 1
    if previus_int % 2 == 0:
        previus_int -= 1

    return next_int, previus_int