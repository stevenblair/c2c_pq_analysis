import csv
import calendar
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator, DateFormatter
import copy
import tables
import shelve
import os.path


plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['ps.useafm'] = True
plt.rcParams['pdf.use14corefonts'] = True


class MonitoringDataProcessing():
    def __init__(self):
        self.MONITORING_VALUES_TIME_INCREMENT = 60 * 5
        self.MONITORING_VALUES_PER_DAY = 24 * 12
        self.MONITORING_VALUES_PER_WEEK = self.MONITORING_VALUES_PER_DAY * 7
        self.CORRELATION_THRESHOLD = 0.9
        self.PERCENTILE = 95
        self.SIMILAR_DEMAND_THRESHOLD_PERCENT = 5.0
        
        self.all_circuit_data = None

        # import all circuit data from file

        with open("circuit_data_input.py") as data_file:
            self.all_circuit_data = eval(data_file.read())
    	self.NOP_labels = [r['NOP_label'] for r in self.all_circuit_data if len(r['NOP_label']) > 0]

        # process NOP log data
        self.NOP_log = []
        if not os.path.isfile('sorted_events'):
	        with open('nop_states_20Aug.log', 'rb') as csvfile:
	            reader = csv.DictReader(csvfile, delimiter='|', skipinitialspace=True)
	            # clean_reader = {k.strip():v.strip() for (k, v) in reader}
	            for row in reader:
	                row2 = {}
	                for k, v in row.iteritems():
	                    # print(k, v)
	                    if v != None:
	                        row2[k.strip()] = v.strip()
	                if len(row2) > 0:
	                    self.NOP_log.append(row2)
	                # print(row['main_desc_1'])

        # data structures
        self.NOP_log_groups = []
        self.sorted_events = {}
        self.circuits_with_week_data = {}
        self.six_day_periods_hours = []
        self.events_with_monitoring_data = {}
        self.monitor_metadata = {}
        self.window = {}
        self.shelf = {}

        # define metrics for power quality analysis
        self.metrics = [
            {'Demand': 'L1_Current_RMS_1_2__1_cyc_Avg_A', 'Voltage': 'L1_N_RMS_1_2__1_cyc_Avg_V', 'THD': 'THD_V_L1_Avg_perc', 'TDD': 'TDD_A_L1_Avg_perc', 'Pst': 'P_st_L1_Avg', 'Plt': 'P_lt_L1_Avg'},
            {'Demand': 'L2_Current_RMS_1_2__1_cyc_Avg_A', 'Voltage': 'L2_N_RMS_1_2__1_cyc_Avg_V', 'THD': 'THD_V_L2_Avg_perc', 'TDD': 'TDD_A_L2_Avg_perc', 'Pst': 'P_st_L2_Avg', 'Plt': 'P_lt_L2_Avg'},
            {'Demand': 'L3_Current_RMS_1_2__1_cyc_Avg_A', 'Voltage': 'L3_N_RMS_1_2__1_cyc_Avg_V', 'THD': 'THD_V_L3_Avg_perc', 'TDD': 'TDD_A_L3_Avg_perc', 'Pst': 'P_st_L3_Avg', 'Plt': 'P_lt_L3_Avg'}
        ]

        # define plot types for power quality metrics
        self.plot_types = [
            {'name': 'Demand', 'index': 'Current_RMS_10_Cycle_Avg_A', 'unit': 'A', 'limits': (0.0, 500.0)},
            {'name': 'RMS L-N voltage', 'index': 'L_N_RMS_10_Cycle_Avg_V', 'unit': 'V', 'limits': (236, 254)},
            {'name': 'Voltage THD', 'index': 'THD_V_Avg_perc', 'unit': '%', 'limits': (0.0, 5.0)},
            {'name': 'TDD', 'index': 'TDD_A_Avg_perc', 'unit': '%', 'limits': (0.0, 20.0)},
            {'name': 'Flicker, Pst', 'index': 'Flicker_P_st_Avg', 'unit': '', 'limits': (0.0, 1.0)},
            {'name': 'Flicker, Plt', 'index': 'Flicker_P_lt_Avg', 'unit': '', 'limits': (0.0, 1.0)},
            # {'name': 'IEC Negative Sequence Current Average (%)', 'index': 'IEC_Negative_Sequence_A_Avg_perc', 'unit': '%', 'limits': (0.0, 200.0)}#,
            # {'name': 'Voltage unbalance', 'index': 'IEC_Negative_Sequence_V_Avg_perc', 'unit': '%', 'limits': (0.0, 1.0)}
        ]


    def run(self, plot_data_overview=True, plot_power_quality_metrics=True):
        if not os.path.isfile('sorted_events'):
	        self.create_event_groups_from_NOP_log()
	        self.create_events()
        else:
            shelf = shelve.open('sorted_events')
            self.sorted_events = shelf['sorted_events']
            shelf.close()
            
        self.find_events_with_week_data(save_pickle=False)

        self.get_all_monitoring_metadata()
        self.print_events()
        
        self.get_percentage_valid_data()
        self.validate_event_date_ranges()
        self.check_for_continuous_data()
        self.validate_freq_sync(find_offsets=True, trusting=True)
        self.test_demand_variation()


        self.compare_metric_combine_phases(metric=['THD_V_L1_Avg_perc', 'THD_V_L2_Avg_perc', 'THD_V_L3_Avg_perc'], xlabel='THD (%)')
        self.compare_metric_combine_phases(metric=['P_st_L1_Avg', 'P_st_L2_Avg', 'P_st_L3_Avg'], xlabel='Flicker (Pst)')
        self.compare_metric(metric=['THD_V_L1_Avg_perc', 'THD_V_L2_Avg_perc', 'THD_V_L3_Avg_perc'], xlabel='THD (%)')
        self.compare_metric(metric=['P_st_L1_Avg', 'P_st_L2_Avg', 'P_st_L3_Avg'], xlabel='Pst')
        self.compare_metric(metric=['P_lt_L1_Avg', 'P_lt_L2_Avg', 'P_lt_L3_Avg'], xlabel='Plt')
        self.compare_metric(metric=['L1_Current_RMS_1_2__1_cyc_Avg_A', 'L2_Current_RMS_1_2__1_cyc_Avg_A', 'L3_Current_RMS_1_2__1_cyc_Avg_A'], xlabel='demand (A)')
        self.compare_all_metrics_csv2()
        self.compare_all_metrics_csv2_for_paper()
        self.compare_all_metrics()
        self.compare_metric(metric=['L1_Current_RMS_1_2__1_cyc_Avg_A', 'L2_Current_RMS_1_2__1_cyc_Avg_A', 'L3_Current_RMS_1_2__1_cyc_Avg_A'], xlabel='Demand (A)')
        self.compare_metric(metric=['TDD_A_L1_Avg_perc', 'TDD_A_L2_Avg_perc', 'TDD_A_L3_Avg_perc'], xlabel='TDD (%))')
        self.compare_metric(metric=['P_st_L1_Avg', 'P_st_L2_Avg', 'P_st_L3_Avg'], xlabel='Flicker st')
        self.neg_seq_hist()

        if plot_data_overview:
            self.visualise_six_day_periods()
            self.visualise_offset()
            self.visualise_has_data()
            self.visualise_freq_sync()
            self.visualise_NOP_state_changes()

        if plot_power_quality_metrics:
            self.visualise_power_quality_metrics(has_two_monitors_only=False)


    def get_label(self, label):
        for l in self.NOP_labels:
            if l in label:
                return l
        return ''

    def get_label_from_ring_ID(self, ring_ID):
        for r in self.all_circuit_data:
            if r['ring_ID'] == ring_ID:
                return r['NOP_label']
        return ''

    def get_primary_name_from_label(self, label):
        for c in self.all_circuit_data:
            if c['NOP_label'] == label:
                return c['ring_ID']
        return ''

    def custom_filter(self, s):
        found_start = False
        start_val = 0.0
        for i, val in enumerate(s):
            if not found_start and val > 0:
                found_start = True
                start = i
                start_val = val
            if found_start and val != start_val:
                end = i + 1
                yield (start, end, start_val)
                found_start = False
                start_val = 0.0
                if val > 0:
                    found_start = True
                    start = i
                    start_val = val
        if found_start:
            yield (start, len(s), start_val)


    def create_event_groups_from_NOP_log(self):
        # organise rows in the NOP log into groups, per C2C primary
        group = []
        label = ''
        for row in self.NOP_log:
            rowLabel = self.get_label(row['main_desc_1'])
            if rowLabel != '':
                if label == rowLabel:
                    group.append(row)
                else:
                    if len(group) > 0:
                        self.NOP_log_groups.append(group)
                    group = [row]
                    label = rowLabel
        # catch any dangling rows
        if len(group) > 0:
            self.NOP_log_groups.append(group)


    def create_events(self):
        events = []

        # parse groups into single events
        for g in self.NOP_log_groups:
            rev_g = reversed(g)
            for i in rev_g:
                if 'Opened' in i['log_state'] or 'Closed' in i['log_state']:
                    # convert to datetime object; ignore fractional second information, if present
                    dt = datetime.datetime.strptime(i['log_date'].title() + ' ' + i['log_time'][:8], '%d-%b-%y %H:%M:%S')
                    events.append({
                        'ring_ID': self.get_primary_name_from_label(self.get_label(i['main_desc_1'])),
                        'state': i['log_state'],
                        # 'date': i['log_date'].title(),
                        # 'time': i['log_time'][:8],  # ignore fractional second information
                        'datetime': dt,
                        'asset': i['main_desc_2'],
                        'sync' : 'False'
                        })

        # sort by primary substation
        for e in events:
            if e['ring_ID'] in self.sorted_events:
                self.sorted_events[e['ring_ID']].append((e['datetime'], e['state'], e['asset']))
            else:
                self.sorted_events[e['ring_ID']] = [(e['datetime'], e['state'], e['asset'])]

        # sort each set of events by date and time
        for ring_ID in self.sorted_events:
            self.sorted_events[ring_ID].sort(key=lambda e: e[0])

        print sum([len(e) for e in self.sorted_events.itervalues()]), 'raw events, across', len([1 for e in self.sorted_events.itervalues() if len(e) > 0]), 'rings'


    def find_events_with_week_data(self, save_pickle=False):
        for ring_ID in sorted(self.sorted_events):
            prev_event_type = ''
            circuit_had_first_close_event = False
            # print ring_ID

            # self.sorted_events[ring_ID][:] = [e for e in self.sorted_events[ring_ID]
            list_copy = copy.deepcopy(self.sorted_events[ring_ID])

            for e in list_copy:
                event_type = e[1]
                if event_type == 'Closed':
                    circuit_had_first_close_event = True

                # remove events which are redundant; only add events after the first 'close' operation
                if event_type == prev_event_type or circuit_had_first_close_event == False:
                    # print 'anomaly:'
                    self.sorted_events[ring_ID].remove(e)

                # print e[0].strftime('%d/%m/%Y, %H:%M:%S:'), event_type, '(' + e[2] + ')'
                prev_event_type = event_type
            prev_event_type = ''

            if not save_pickle:
                start_date = datetime.datetime(2010, 4, 1)
                for i, event in enumerate(self.sorted_events[ring_ID]):
                    date_delta_before = event[0] - start_date
                    start_date = event[0]
                    if i + 1 < len(self.sorted_events[ring_ID]):
                        date_delta_after = self.sorted_events[ring_ID][i + 1][0] - start_date
                        if date_delta_before.days >= 7 and date_delta_after.days >= 7:
                            if ring_ID in self.circuits_with_week_data:
                                self.circuits_with_week_data[ring_ID].append((start_date, event[1]))
                            else:
                                self.circuits_with_week_data[ring_ID] = [(start_date, event[1])]
                            # print ring_ID, 'ok'
                        elif date_delta_before.days >= 7 and date_delta_after.days >= 6:
                            self.six_day_periods_hours.append(24.0 - float(date_delta_after.seconds) / (60.0 * 60.0))
                        elif date_delta_before.days >= 6 and date_delta_after.days >= 7:
                            self.six_day_periods_hours.append(24.0 - float(date_delta_before.seconds) / (60.0 * 60.0))
                        elif date_delta_before.days >= 6 and date_delta_after.days >= 6:
                            pass
                            #self.six_day_periods_hours.append(24.0 - (float(date_delta_before.seconds) / (60.0 * 60.0) + float(date_delta_after.seconds) / (60.0 * 60.0)) / 2)
            else:
                shelf = shelve.open('sorted_events')
                shelf['sorted_events'] = self.sorted_events
                shelf.close()
                
                # print '    ', str(e[0].time) + ', ' + str(e[0].day) + '/' + str(e[0].month) + '/' + str(e[0].year), e[1]

        print len(self.circuits_with_week_data), 'circuits are suitable for analysis, totaling', str(sum([len(events) for events in self.circuits_with_week_data.itervalues()])), 'NOP state changes'
        # print len(self.six_day_periods_hours), 'events are slightly too short', min(self.six_day_periods_hours), max(self.six_day_periods_hours), sum(self.six_day_periods_hours) / len(self.six_day_periods_hours)
        # for k in self.circuits_with_week_data:
        #     print k, self.circuits_with_week_data[k]


    def visualise_six_day_periods(self):
        fig = plt.figure(figsize=(8, 4.5), facecolor='w')
        plt.hist(self.six_day_periods_hours, bins=96, histtype='stepfilled', color='r', alpha=0.6, label='Uniform', cumulative=False)
        # plt.boxplot(self.six_day_periods_hours, vert=False)
        plt.xlabel("Time missing (hours)")
        plt.ylabel("Number of occurrences")
        plt.tight_layout()

        # plt.show()
        plt.savefig('plots\\visualise_six_day_periods.png', dpi=200)


    def get_all_monitoring_metadata(self):
        f = tables.open_file('data\monitoring-data-float32-no-compression.h5', mode='r+', driver="H5FD_CORE")
        monitor_tables = f.root
        # print len(monitor_tables)
        # print monitor_tables
        for node in monitor_tables:
            # print node._v_title, node._v_nchildren, node._f_get_child('readout')
            self.monitor_metadata[node._v_title] = {
                'monitor_name': node._v_title,
                'table': node._f_get_child('readout'),
                'ring_ID': node._v_attrs.ring_ID,
                'primary_name': node._v_attrs.primary_name,
                'earliest_date': datetime.datetime.utcfromtimestamp(node._v_attrs.earliest_date),
                'latest_date': datetime.datetime.utcfromtimestamp(node._v_attrs.latest_date),
            }
        # table = f.root.monitor_1.readout
        # rows = [row for row in table.where('ring_ID == 69')]
        # print len(rows)



    def find_monitors_from_ring_ID(self, ring_ID):
        matches = []

        # ignore additional monitoring devices on fully-monitored ring circuit; keep only the two devices nearest NOP
        for c in self.all_circuit_data:
            if 'NOP_monitors' in c:
                NOP_monitors = c['NOP_monitors']

        # find exact matches
        rows = [m for m in self.monitor_metadata.itervalues() if m['ring_ID'] == ring_ID]
        # rows = [row for row in table.where('ring_ID == ' + str(ring_ID))]

        for r in rows:
            if ring_ID == r['ring_ID']:
                if ring_ID == 58 and r['primary_name'] not in NOP_monitors:
                    pass
                else:
                    matches.append(r)

        return matches



# def find_earliest_monitor_date(monitor_name):
#     # latest_date2 = datetime.datetime.utcfromtimestamp(table.cols.date[table.colindexes['date'][-1]])
#     table = monitor_tables[monitor_name]
#     earliest_date = datetime.datetime.utcfromtimestamp(min(row['date'] for row in table.where('monitor_name == \'' + str(monitor_name) + '\'')))
#     return earliest_date

# def find_largest_monitor_date(monitor_name):
#     table = monitor_tables[monitor_name]
#     latest_date = datetime.datetime.utcfromtimestamp(max(row['date'] for row in table.where('monitor_name == \'' + str(monitor_name) + '\'')))
#     return latest_date


    def ring_name_from_ID(self, ring_ID):
        for m in self.monitor_metadata.itervalues():
            if m['ring_ID'] == ring_ID:
                return m['primary_name'].title()




    def get_percentage_valid_data(self):
        # determine percentage of complete data between start and end points
        for m in self.monitor_metadata.itervalues():
            table = m['table']
            values_percent_range = 100 * table.nrows / ((m['latest_date'] - m['earliest_date']).total_seconds() / (self.MONITORING_VALUES_TIME_INCREMENT))
            m['values_percent_range'] = values_percent_range
            print m['primary_name'], m['monitor_name'], table.nrows, '{:.2f}'.format(values_percent_range) + '%'



    def date_range_by_day(self, start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + datetime.timedelta(n)


    def mean(self, x):
        # assert len(x) > 0
        return float(sum(x)) / len(x)

    def corr(self, x, y):
        # assert len(x) == len(y)
        n = len(x)
        # assert n > 0
        avg_x = self.mean(x)
        avg_y = self.mean(y)
        diffprod = 0
        xdiff2 = 0
        ydiff2 = 0
        for i in range(n):
            xdiff = x[i] - avg_x
            ydiff = y[i] - avg_y
            diffprod += xdiff * ydiff
            xdiff2 += xdiff * xdiff
            ydiff2 += ydiff * ydiff

        return diffprod / math.sqrt(xdiff2 * ydiff2)

    def print_monitoring_data_summary(self, prefix):
        print prefix + ':', len(self.events_with_monitoring_data), 'NOP state changes with monitoring data, from', \
            len(set([m['monitor_name'] for v in self.events_with_monitoring_data.itervalues() for m in v])), 'monitoring devices and', \
            len(set([m['primary_name'] for v in self.events_with_monitoring_data.itervalues() for m in v])), 'rings'

    def validate_event_date_ranges(self):
        for ring_ID in self.circuits_with_week_data:
            monitors = self.find_monitors_from_ring_ID(ring_ID)
            # print ring_ID, ring_name_from_ID(ring_ID), len(monitors)
            if len(monitors) > 0:
                for event, event_type in self.circuits_with_week_data[ring_ID]:
                    event_before = event - datetime.timedelta(days=7)
                    event_after = event + datetime.timedelta(days=7)
                    for monitor in monitors:
                        # print earliest_date, latest_date
                        # check date range is valid
                        if monitor['earliest_date'] <= event_before and monitor['latest_date'] >= event_after:
                            if (ring_ID, event, event_type) in self.events_with_monitoring_data:
                                self.events_with_monitoring_data[(ring_ID, event, event_type)].append(monitor)
                            else:
                                self.events_with_monitoring_data[(ring_ID, event, event_type)] = [monitor]
                            # print '  within monitored date range for NOP change by', monitor['primary_name'], 'on', event

        self.print_monitoring_data_summary('validate_event_date_ranges()')


    def check_for_continuous_data(self):
        for ring_ID, event, event_type in self.events_with_monitoring_data.copy():
            monitors = list(self.events_with_monitoring_data[(ring_ID, event, event_type)])
            for m in monitors:
                start_datetime = event - datetime.timedelta(days=7)
                end_datetime = event + datetime.timedelta(days=7)
                table = m['table']

                values_before = [row['date'] for row in table.where('(date >= ' + str(calendar.timegm(start_datetime.timetuple())) + ') & (date <= ' + str(calendar.timegm(event.timetuple())) + ')')]
                values_after = [row['date'] for row in table.where('(date > ' + str(calendar.timegm(event.timetuple())) + ') & (date <= ' + str(calendar.timegm(end_datetime.timetuple())) + ')')]
                
                # remove from list
                if len(values_before) != self.MONITORING_VALUES_PER_WEEK or len(values_after) != self.MONITORING_VALUES_PER_WEEK:
                    print 'removing due to missing data:', self.ring_name_from_ID(m['ring_ID']), m['monitor_name'].title(), event_type, len(values_before), len(values_after)
                    self.events_with_monitoring_data[ring_ID, event, event_type].remove(m)

            # if there's no valid monitoring data remaining, discard this event
            if len(self.events_with_monitoring_data[ring_ID, event, event_type]) == 0:
                del self.events_with_monitoring_data[(ring_ID, event, event_type)]

        self.print_monitoring_data_summary('check_for_continuous_data()')


    def test_time_offset(self, monitor, day, offset, column_mean):
        new_day = day + offset
        day_end = new_day + datetime.timedelta(1)
        query = '(date >= ' + str(calendar.timegm(new_day.timetuple())) + ') & (date < ' + str(calendar.timegm(day_end.timetuple())) + ')'

        # get raw frequency data
        if new_day >= monitor['earliest_date'] and new_day <= monitor['latest_date']:
            table = monitor['table']
            # it is assumed that the table is already sorted by data in ascending order
            values = [row['Frequency__1_cyc_Avg_Hz'] for row in table.where(query)]
            if len(values) == self.MONITORING_VALUES_PER_DAY:
                corr = self.corr(values, column_mean)
                if corr >= self.CORRELATION_THRESHOLD:
                    return True
        return False


    def find_time_offset_direction(self, monitor, day, column_mean, direction):
        corr = 0.0
        iterations = 0
        offset = datetime.timedelta(0)

        # use a search space with increments of 5 minutes,
        # with a maximum difference of 48 hours
        while corr < self.CORRELATION_THRESHOLD and iterations < 48*12:
            iterations += 1
            offset = offset + direction * datetime.timedelta(seconds=60*5)
            new_day = day + offset
            day_end = new_day + datetime.timedelta(1)
            query = '(date >= ' + str(calendar.timegm(new_day.timetuple())) + ') & (date < ' + str(calendar.timegm(day_end.timetuple())) + ')'

            # get raw frequency data
            if new_day >= monitor['earliest_date'] and new_day <= monitor['latest_date']:
                table = monitor['table']
                # it is assumed that the table is already sorted by data in ascending order
                values = [row['Frequency__1_cyc_Avg_Hz'] for row in table.where(query)]
                if len(values) == self.MONITORING_VALUES_PER_DAY:
                    corr = self.corr(values, column_mean)
                    if corr >= self.CORRELATION_THRESHOLD:
                        return offset
                    # monitors_in_window.append({'monitor': monitor, 'values': values})
        return datetime.timedelta(0)

    def find_time_offset(self, monitor, day, column_mean):
        offset = self.find_time_offset_direction(monitor, day, column_mean, -1)
        if offset == datetime.timedelta(0):
            offset = self.find_time_offset_direction(monitor, day, column_mean, 1)
        return offset

    def validate_freq_sync(self, find_offsets=True, trusting=True):
        # delete the local copies of these files to force a refresh the data
        if find_offsets and trusting:
            self.shelf = shelve.open('validate_freq_sync_shelf_with_trusting')
        elif find_offsets:
            self.shelf = shelve.open('validate_freq_sync_shelf_with_offsets')
        else:
            self.shelf = shelve.open('validate_freq_sync_shelf_no_offsets')

        # build data structure of all freq measurements in day window
        if 'window' in self.shelf:
            self.window = self.shelf['window']
        else:
            self.window = {}

            start = min([m['earliest_date'] for m in self.monitor_metadata.itervalues()])
            start = start.replace(hour=0, minute=0, second=0)   # set to midnight, at start of day
            end = max([m['latest_date'] for m in self.monitor_metadata.itervalues()]) + datetime.timedelta(1)
            print 'max date range:', start, end

            for day in self.date_range_by_day(start, end):
                self.window[day] = []
                monitors_in_window = []
                day_end = day + datetime.timedelta(1)
                # query = '(date >= ' + str(calendar.timegm(day.timetuple())) + ') & (date < ' + str(calendar.timegm(day_end.timetuple())) + ')'

                # get raw frequency data
                for m in self.monitor_metadata.itervalues():
                    if day >= m['earliest_date'] and day <= m['latest_date']:
                        query = '(date >= ' + str(calendar.timegm(day.timetuple())) + ') & (date < ' + str(calendar.timegm(day_end.timetuple())) + ')'
                        # adjust window based on previous day's offset for this monitor
                        day_prev = day - datetime.timedelta(1)
                        offset = datetime.timedelta(0)
                        if day_prev in self.window:
                            if find_offsets:
                                offset = next((x['offset'] for x in self.window[day_prev] if x['monitor_name'] == m['monitor_name'] and x['valid']), datetime.timedelta(0))
                                if offset != datetime.timedelta(0):
                                    query = '(date >= ' + str(calendar.timegm((day + offset).timetuple())) + ') & (date < ' + str(calendar.timegm((day_end + offset).timetuple())) + ')'
                                    # print 'adjusting window:', m['primary_name'], m['monitor_name'], offset

                        table = m['table']
                        # it is assumed that the table is already sorted by data in ascending order
                        values = [row['Frequency__1_cyc_Avg_Hz'] for row in table.where(query)]
                        if len(values) == self.MONITORING_VALUES_PER_DAY:
                            monitors_in_window.append({'monitor': m, 'values': values, 'offset': offset})
                        else:
                            # print 'rejected', m['primary_name'], m['monitor_name'], day
                            self.window[day].append({'primary_name': m['primary_name'], 'monitor_name': m['monitor_name'],
                                'valid': False, 'corr': 0.0, 'offset': datetime.timedelta(0)})
                    else:
                        self.window[day].append({'primary_name': m['primary_name'], 'monitor_name': m['monitor_name'],
                            'valid': False, 'corr': 0.0, 'offset': datetime.timedelta(0)})
                        # if day >= datetime.datetime(2013, 06, 27) and day < datetime.datetime(2013, 06, 28):
                        #     if len(values) > 5:
                        #         print m['monitor_name'], values[6]

                # print len(monitors_in_window)
                if len(monitors_in_window) > 0:
                    column_mean = []
                    # column_sum = []
                    for i in range(0, self.MONITORING_VALUES_PER_DAY):
                        column_sum = 0.0
                        for m in monitors_in_window:
                            column_sum += m['values'][i]
                        # column_sum.append(column_sum)
                        column_mean.append(column_sum / len(monitors_in_window))

                    # if day >= datetime.datetime(2013, 06, 4) and day < datetime.datetime(2013, 06, 5):
                    #     print 'mean,', [m['monitor']['monitor_name'] for m in monitors_in_window]
                    #     for i, mean in enumerate(column_mean):
                    #         print 'mean', str(mean) + ',', [m['values'][i] for m in monitors_in_window]

                    for m in monitors_in_window:
                        if len(monitors_in_window) >= 20:
                            corr = self.corr(m['values'], column_mean)
                            if corr < self.CORRELATION_THRESHOLD:
                                if find_offsets:
                                    retest_offset = True
                                    if day - datetime.timedelta(1) in self.window:
                                        prev = next((x for x in self.window[day - datetime.timedelta(1)] if x['monitor_name'] == m['monitor']['monitor_name'] and x['valid']), None)
                                        if prev is not None:
                                            offset_prev = prev['offset']
                                            if self.test_time_offset(m['monitor'], day, offset_prev, column_mean) == True:
                                                self.window[day].append({'primary_name': m['monitor']['primary_name'], 'monitor_name': m['monitor']['monitor_name'],
                                                    'valid': True, 'corr': corr, 'offset': offset_prev})
                                                retest_offset = False

                                    if retest_offset:
                                        offset = self.find_time_offset(m['monitor'], day, column_mean)
                                        # print 'offset:', offset, m['monitor']['primary_name'], m['monitor']['monitor_name']
                                        if offset == datetime.timedelta(0):
                                            self.window[day].append({'primary_name': m['monitor']['primary_name'], 'monitor_name': m['monitor']['monitor_name'],
                                                'valid': False, 'corr': corr, 'offset': datetime.timedelta(0)})
                                        else:
                                            self.window[day].append({'primary_name': m['monitor']['primary_name'], 'monitor_name': m['monitor']['monitor_name'],
                                                'valid': True, 'corr': corr, 'offset': offset})
                                else:
                                    self.window[day].append({'primary_name': m['monitor']['primary_name'], 'monitor_name': m['monitor']['monitor_name'],
                                        'valid': False, 'corr': corr, 'offset': datetime.timedelta(0)})
                            else:
                                offset = datetime.timedelta(0)
                                if 'offset' in m:
                                    offset = m['offset']
                                self.window[day].append({'primary_name': m['monitor']['primary_name'], 'monitor_name': m['monitor']['monitor_name'],
                                    'valid': True, 'corr': corr, 'offset': offset})
                                # window[day] = {'number_invalid': len(monitors_in_window), 'monitors': monitors_in_window}
                                # print(day, m['monitor']['primary_name'], m['monitor']['monitor_name'])
                                # results.append(corr)
                        else:
                            offset = datetime.timedelta(0)
                            if 'offset' in m:
                                offset = m['offset']
                            self.window[day].append({'primary_name': m['monitor']['primary_name'], 'monitor_name': m['monitor']['monitor_name'],
                                    'valid': False, 'corr': 0.0, 'offset': offset})

                # print results
                if len(self.window[day]) == 0:
                    print day, '', len(self.window[day])
                else:
                    print day, '', sum([1 for w in self.window[day] if w['valid'] == True]), 'out of', len(monitors_in_window), 'total of', len(self.window[day])


            # review invalid gaps and trust time sync if freq corr valid for on either side of gap
            if trusting:
                for day in self.date_range_by_day(start, end):
                    for m in self.window[day]:
                        # check that previous day was valid to avoid over-zealous trusting at start of trial data
                        day_prev = day - datetime.timedelta(1)
                        if day_prev in self.window:
                            day_prev_valid = next((x['valid'] for x in self.window[day_prev] if x['monitor_name'] == m['monitor_name'] and x['valid']), False)
                            if m['valid'] == False and day_prev_valid:
                                next_valid_day = day + datetime.timedelta(1)
                                for day_next in self.date_range_by_day(next_valid_day, next_valid_day + datetime.timedelta(7)):
                                    if day_next in self.window:
                                        next_valid = next((x2['valid'] for x2 in self.window[day_next] if x2['monitor_name'] == m['monitor_name'] and x2['valid']), False)
                                        if next_valid:
                                            # print 'trusting', m['monitor_name'], day, day_next
                                            for day_change in self.date_range_by_day(day, day_next):
                                                monitor_change = next((x3 for x3 in self.window[day_change] if x3['monitor_name'] == m['monitor_name']), None)
                                                monitor_change['valid'] = True
                                            break


            self.shelf['window'] = self.window
            # shelf.close()


        # reject locations with invalid freq sync
        for ring_ID, event, event_type in self.events_with_monitoring_data.copy():
            monitors = list(self.events_with_monitoring_data[(ring_ID, event, event_type)])
            for m in monitors:
                start_datetime = event - datetime.timedelta(days=7)
                end_datetime = event + datetime.timedelta(days=7)

                valid = True
                for day in self.date_range_by_day(start_datetime, end_datetime):
                    day_index = day.replace(hour=0, minute=0, second=0)   # set to midnight, at start of day
                    if day_index in self.window:
                        for m2 in self.window[day_index]:
                            if m2['monitor_name'] == m['monitor_name'] and m2['valid'] == False:
                                valid = False
                    else:
                        print 'defaulting invalid', day_index
                        valid = False

                # remove from list
                if valid == False:
                    print 'removing due to clock sync:', self.ring_name_from_ID(m['ring_ID']), m['monitor_name'].title(), event_type
                    self.events_with_monitoring_data[(ring_ID, event, event_type)].remove(m)

            # if there's no valid monitoring data remaining, discard this event
            if len(self.events_with_monitoring_data[(ring_ID, event, event_type)]) == 0:
                del self.events_with_monitoring_data[(ring_ID, event, event_type)]

        self.print_monitoring_data_summary('validate_freq_sync()')


    def has_non_zero_offset(self, monitor_name, event_date):
        # assumes offset is constant for period of interest
        day = event_date.replace(hour=0, minute=0, second=0)   # set to midnight, at start of day
        zero_offset = datetime.timedelta(0)

        if day in self.window:
            for day_offset in self.window[day]:
                if monitor_name == day_offset['monitor_name'] and day_offset['offset'] != zero_offset and day_offset['valid']:
                    return True
        else:
            print 'invalid day:', day, event_date

        return False

    def get_offset(self, monitor_name, event_date):
        # assumes offset is constant for period of interest
        day = event_date.replace(hour=0, minute=0, second=0)   # set to midnight, at start of day

        for day_offset in self.window[day]:
            if monitor_name == day_offset['monitor_name'] and day_offset['valid']:
                # print 'offset found:', day_offset['offset']
                return day_offset['offset']

        # print 'offset not found'
        return datetime.timedelta(0)


    def visualise_offset(self):
        datemin = datetime.date(2013, 2, 1)
        datemax = datetime.date(2014, 8, 1)
        fig = plt.figure(figsize=(30, 50), facecolor='w')

        axisNum = 0
        prev_primary_name = ''
        y_label_colours = ('#333333', '#000000')
        y_label_colour_index = 0

        has_data_x = {}
        has_data_y = {}
        batches = {}

        # for day,m in self.window.itervalues():
        #     if (m['primary_name'], m['monitor_name']) not in has_data_x:
        #         has_data_x[(m['primary_name'], m['monitor_name'])] = []
        #     if (m['primary_name'], m['monitor_name']) not in has_data_y:
        #         has_data_y[(m['primary_name'], m['monitor_name'])] = []


        for m in self.monitor_metadata.itervalues():
            if (m['primary_name'], m['monitor_name']) not in has_data_x:
                has_data_x[(m['primary_name'], m['monitor_name'])] = []
            if (m['primary_name'], m['monitor_name']) not in has_data_y:
                has_data_y[(m['primary_name'], m['monitor_name'])] = []

            start_datetime = m['earliest_date']
            end_datetime = m['latest_date']
            # for day in self.date_range_by_day(start_datetime, end_datetime):
            #     if day in self.window:
            for day in sorted(self.window.keys()):
                offset = next((x['offset'] for x in self.window[day] if x['monitor_name'] == m['monitor_name']), datetime.timedelta(0))

                has_data_x[(m['primary_name'], m['monitor_name'])].append(day)
                has_data_y[(m['primary_name'], m['monitor_name'])].append(offset.total_seconds())

                # has_data_batch[(m['primary_name'], m['monitor_name'], day)] = data[0]

                # if len(data) > 0 and data[0] not in [batches[(primary, mon, d)] for primary, mon, d in batches if mon == m['monitor_name']]:
                #     # print 'adding batch'
                #     batches[(m['primary_name'], m['monitor_name'], day)] = data[0]

        y_limits = [min([v for k,v in has_data_y.iteritems()]), max([v for k,v in has_data_y.iteritems()])]
        # print has_data_y.values()
        # print 'y_limits', y_limits

        for n,p in sorted(has_data_y.keys()):
            axisNum += 1
            if prev_primary_name != n:
                prev_primary_name = n
                y_label_colour_index = 1 if y_label_colour_index == 0 else 0

            ax = plt.subplot(len(has_data_y), 1, axisNum)
            x2num = mdates.date2num(has_data_x[(n,p)])
            # plt.plot(x2num, has_data_y[(n,p)], color='g')
            plt.step(x2num, has_data_y[(n,p)], where='post', color='g')
            # ax.fill_between(x2num, 0.0, has_data_y[(n,p)], facecolor='green', alpha=0.5, interpolate=False)
            fill_colour = (128.0 / 255.0, 191.0 / 255.0, 128.0 / 255.0)
            for start, end, val in self.custom_filter(has_data_y[(n,p)]):
                mask = np.zeros_like(has_data_y[(n,p)])
                mask[start: end] = val
                ax.fill_between(x2num, 0.0, val, where=mask, facecolor=fill_colour, edgecolor=fill_colour)
            ax.format_xdata = mdates.DateFormatter('%d/%m/%Y, %H:%M:%S')
            ax.xaxis.set_major_locator(MonthLocator(range(1,13), bymonthday=1, interval=1))
            ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))

            ax.set_yticks([])   # hide y tick labels
            ax.tick_params(axis='x', which='major', color='#AAAAAA', labelsize=12, zorder=-10)
            ax.tick_params(axis='y', which='major', labelsize=8)
            ax.set_xlim(datemin, datemax)
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.xaxis.set_ticks_position('bottom')
            plt.ylim([-37000, 3700])
            # plt.ylim(y_limits)
            ax.set_ylabel(n.title() + ', ' + p.title(), rotation=0, fontsize=8, ha='right', va='center')
            ax.yaxis.label.set_color(y_label_colours[y_label_colour_index])

            batches_monitors = [(day, batches[(primary, mon, day)]) for primary, mon, day in batches if mon == p]
            for b in batches_monitors:
                plt.axvline(b[0], color='r', alpha=0.8, linestyle='-', lw=2)

        fig.autofmt_xdate()
        plt.subplots_adjust(left=0.115, bottom=0.04, right=0.99, top=0.99, wspace=0.20, hspace=0.21)
        # plt.tight_layout()

        # plt.show()
        # plt.savefig('plots\\visualise_offset.png', dpi=300)
        plt.savefig('plots\\visualise_offset.pdf')


    def visualise_has_data(self):
        datemin = datetime.date(2013, 2, 1)
        datemax = datetime.date(2014, 8, 1)
        fig = plt.figure(figsize=(30, 40), facecolor='w')
        axisNum = 0

        # get count of values per day, and divide by expected amount
        # shelf = shelve.open('validate_freq_sync_shelf')
        if 'has_data_x' in self.shelf and 'has_data_y' in self.shelf and 'batches' in self.shelf:
            has_data_x = self.shelf['has_data_x']
            has_data_y = self.shelf['has_data_y']
            batches = self.shelf['batches']
        else:
            has_data_x = {}
            has_data_y = {}
            batches = {}
            for m in self.monitor_metadata.itervalues():
                if (m['primary_name'], m['monitor_name']) not in has_data_x:
                    has_data_x[(m['primary_name'], m['monitor_name'])] = []
                if (m['primary_name'], m['monitor_name']) not in has_data_y:
                    has_data_y[(m['primary_name'], m['monitor_name'])] = []

                table = m['table']
                start_datetime = m['earliest_date']
                end_datetime = m['latest_date']
                for day in self.date_range_by_day(start_datetime, end_datetime):
                    day_end = day + datetime.timedelta(1)
                    query = '(date >= ' + str(calendar.timegm(day.timetuple())) + ') & (date < ' + str(calendar.timegm(day_end.timetuple())) + ')'
                    data = [row['batch'] for row in table.where(query)]
                    nrows = len(data)
                    values_percent_range = float(nrows) / float(self.MONITORING_VALUES_PER_DAY)
                    if values_percent_range > 1.0:
                        print 'excessive day data:', m['monitor_name'], day
                    m['values_percent_range'] = values_percent_range
                    # print m['primary_name'], m['monitor_name'], table.nrows, '{:.1f}'.format(values_percent_range) + '%'
                    has_data_x[(m['primary_name'], m['monitor_name'])].append(day)
                    has_data_y[(m['primary_name'], m['monitor_name'])].append(values_percent_range)
                    # has_data_batch[(m['primary_name'], m['monitor_name'], day)] = data[0]

                    if len(data) > 0 and data[0] not in [batches[(primary, mon, d)] for primary, mon, d in batches if mon == m['monitor_name']]:
                        # print 'adding batch'
                        batches[(m['primary_name'], m['monitor_name'], day)] = data[0]

            self.shelf['has_data_x'] = has_data_x
            self.shelf['has_data_y'] = has_data_y
            self.shelf['batches'] = batches

        prev_primary_name = ''
        y_label_colours = ('#333333', '#000000')
        y_label_colour_index = 0

        for n,p in sorted(has_data_y.keys()):
            axisNum += 1
            if prev_primary_name != n:
                prev_primary_name = n
                y_label_colour_index = 1 if y_label_colour_index == 0 else 0

            ax = plt.subplot(len(has_data_y), 1, axisNum)

            x2num = mdates.date2num(has_data_x[(n,p)])
            # plt.plot(x2num, has_data_y[(n,p)], color='g')
            # plt.step(x2num, has_data_y[(n,p)], where='post', color='g')
            # ax.fill_between(x2num, 0.0, has_data_y[(n,p)], facecolor='green', alpha=0.5, interpolate=False)
            fill_colour_g = (128.0 / 255.0, 191.0 / 255.0, 128.0 / 255.0)#(0, 128.0 / 255.0, 0.0, 0.5)
            fill_colour_r = (1.0, 179.0 / 255.0, 179.0 / 255.0)#(244.0 / 255.0, 0, 0, 0.5)
            for start, end, val in self.custom_filter(has_data_y[(n,p)]):
                mask = np.zeros_like(has_data_y[(n,p)])
                mask[start: end] = val
                ax.fill_between(x2num, 0.0, val, where=mask, facecolor=fill_colour_g if val == 1.0 else fill_colour_r, edgecolor=fill_colour_g if val == 1.0 else fill_colour_r)
            ax.format_xdata = mdates.DateFormatter('%d/%m/%Y, %H:%M:%S')
            ax.xaxis.set_major_locator(MonthLocator(range(1,13), bymonthday=1, interval=1))
            ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))

            ax.set_yticks([])   # hide y tick labels
            ax.tick_params(axis='x', which='major', color='#AAAAAA', labelsize=14, zorder=-10)
            ax.tick_params(axis='y', which='major', labelsize=8)
            ax.set_xlim(datemin, datemax)
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.xaxis.set_ticks_position('bottom')
            plt.ylim([0.0, 1.1])
            ax.set_ylabel(n.title() + '\n' + p.title(), rotation=0, fontsize=12, ha='right', va='center')
            # ax.yaxis.label.set_color(y_label_colours[y_label_colour_index])

            batches_monitors = [(day, batches[(primary, mon, day)]) for primary, mon, day in batches if mon == p]
            for b in batches_monitors:
                plt.axvline(b[0], color='b', alpha=0.8, linestyle='-', lw=2)

        fig.autofmt_xdate()
        plt.subplots_adjust(left=0.10, bottom=0.02, right=0.995, top=0.995, wspace=0.20, hspace=0.21)
        # plt.tight_layout()

        # plt.show()
        plt.savefig('plots\\visualise_has_data.png', dpi=300)
        plt.savefig('plots\\visualise_has_data.pdf')

        # shelf.close()


    def visualise_freq_sync(self):
        import matplotlib.cm as cm
        # shelf = shelve.open('validate_freq_sync_shelf')

        datemin = datetime.date(2013, 2, 1)
        datemax = datetime.date(2014, 8, 1)
        fig = plt.figure(figsize=(30, 40), facecolor='w')
        axisNum = 0

        x = {}
        y = {}
        for day in sorted(self.window.keys()):
            for m in self.window[day]:
                if (m['primary_name'], m['monitor_name']) not in x:
                    x[(m['primary_name'], m['monitor_name'])] = []
                if (m['primary_name'], m['monitor_name']) not in y:
                    y[(m['primary_name'], m['monitor_name'])] = []
                x[(m['primary_name'], m['monitor_name'])].append(day + m['offset'])
                y[(m['primary_name'], m['monitor_name'])].append(1.0 if m['valid'] == True else m['corr'])

        prev_primary_name = ''
        y_label_colours = ('#333333', '#000000')
        y_label_colour_index = 0

        for n,p in sorted(y.keys()):
            axisNum += 1
            if prev_primary_name != n:
                prev_primary_name = n
                y_label_colour_index = 1 if y_label_colour_index == 0 else 0

            ax = plt.subplot(len(y), 1, axisNum)
            x2num = mdates.date2num(x[(n,p)])

            # plt.step(x2num, y[(n,p)], where='post', color='g')
            fill_colour = (128.0 / 255.0, 191.0 / 255.0, 128.0 / 255.0)
            for start, end, val in self.custom_filter(y[(n,p)]):
                fill_colour = (128.0 / 255.0, 191.0 / 255.0, 128.0 / 255.0) if val == 1.0 else (1.0, 179.0 / 255.0, 179.0 / 255.0)
                mask = np.zeros_like(y[(n,p)])
                mask[start: end] = val
                ax.fill_between(x2num, 0.0, val, where=mask, facecolor=fill_colour, edgecolor=fill_colour)
                # ax.fill_between(x2num, 0.0, val, where=mask, facecolor='g', alpha=0.5)

            # bar_width = x2num[1] - x2num[0]
            # for pointx, pointy in zip(x2num, y[(n,p)]):
            #     current_color = cm.RdYlGn(pointy)
            #     plt.bar(pointx, pointy, bar_width, color=current_color, linewidth=0)

            ax.format_xdata = mdates.DateFormatter('%d/%m/%Y, %H:%M:%S')
            ax.xaxis.set_major_locator(MonthLocator(range(1,13), bymonthday=1, interval=1))
            ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))

            if 'batches' in self.shelf:
                batches = self.shelf['batches']
                batches_monitors = [(day, batches[(primary, mon, day)]) for primary, mon, day in batches if mon == p]
                for b in batches_monitors:
                    plt.axvline(b[0], color='b', alpha=0.8, linestyle='-', lw=2)

            ax.set_yticks([])   # hide y tick labels
            ax.tick_params(axis='x', which='major', color='#AAAAAA', labelsize=14, zorder=-10)
            ax.tick_params(axis='y', which='major', labelsize=8)
            ax.set_xlim(datemin, datemax)
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.xaxis.set_ticks_position('bottom')
            plt.ylim([0.0, 1.1])
            ax.set_ylabel(n.title() + '\n' + p.title(), rotation=0, fontsize=12, ha='right', va='center')
            # ax.yaxis.label.set_color(y_label_colours[y_label_colour_index])

        fig.autofmt_xdate()
        plt.subplots_adjust(left=0.10, bottom=0.02, right=0.995, top=0.995, wspace=0.20, hspace=0.21)
        # plt.tight_layout()

        # plt.show()
        plt.savefig('plots\\visualise_freq_sync.png', dpi=300)
        plt.savefig('plots\\visualise_freq_sync.pdf')


    def print_events(self):
        print 'Ring ID,Primary name,Date and time,Event type,NOP log label'

        for ring_ID in self.circuits_with_week_data:
            for event, event_type in sorted(self.circuits_with_week_data[ring_ID], key=lambda k: k[0]):
                print str(ring_ID) + ',' + self.ring_name_from_ID(ring_ID) + ',' + str(event) + ',' + 'NOP ' + event_type + ',' + self.get_label_from_ring_ID(ring_ID)

        # for ring_ID, event, event_type in sorted(self.events_with_monitoring_data, key=lambda k: k[1]):
        #     print str(ring_ID) + ',' + self.ring_name_from_ID(ring_ID) + ',' + str(event) + ',' + 'NOP ' + event_type + ',' + self.get_label_from_ring_ID(ring_ID)

    def test_demand_variation(self):
        phase_closed = [[], [], []]
        phase_open = [[], [], []]

        demand_percentile_diff = []

        for ring_ID, event, event_type in self.events_with_monitoring_data.copy():
            monitors = list(self.events_with_monitoring_data[(ring_ID, event, event_type)])
            for monitor in monitors:
            # monitor = monitors[0]
                table = monitor['table']
                event_with_offset = event + self.get_offset(monitor['monitor_name'], event)
                start_datetime = event_with_offset - datetime.timedelta(days=7)
                end_datetime = event_with_offset + datetime.timedelta(days=7)

                metrics = ['L1_Current_RMS_1_2__1_cyc_Avg_A', 'L2_Current_RMS_1_2__1_cyc_Avg_A', 'L3_Current_RMS_1_2__1_cyc_Avg_A']

                values_before = []
                values_after = []
                for i, metric in enumerate(metrics):
                    values_before.append([row[metric] for row in table.where('(date >= ' + str(calendar.timegm(start_datetime.timetuple())) + ') & (date <= ' + str(calendar.timegm(event_with_offset.timetuple())) + ')')])
                    values_after.append([row[metric] for row in table.where('(date > ' + str(calendar.timegm(event_with_offset.timetuple())) + ') & (date <= ' + str(calendar.timegm(end_datetime.timetuple())) + ')')])
                values_closed = values_after if event_type == 'Closed' else values_before
                values_open = values_before if event_type == 'Closed' else values_after

                for i, phase_open, phase_closed in zip(range(1, 4), values_open, values_closed):
                    # print i, np.percentile(phase_open, self.PERCENTILE), np.percentile(phase_closed, self.PERCENTILE)
                    phase_open_mean = np.mean(phase_open)#, self.PERCENTILE)
                    phase_closed_mean = np.mean(phase_closed)#, self.PERCENTILE)
                    percent_diff = 100.0 * (phase_open_mean - phase_closed_mean) / phase_open_mean
                    if np.abs(percent_diff) < self.SIMILAR_DEMAND_THRESHOLD_PERCENT:
                        demand_percentile_diff.append(percent_diff)
                    else:
                        # remove monitor
                        if monitor in self.events_with_monitoring_data[(ring_ID, event, event_type)]:
                            print 'removing monitor due to demand difference:', self.ring_name_from_ID(monitor['ring_ID']), monitor['monitor_name'].title(), event_type
                            self.events_with_monitoring_data[(ring_ID, event, event_type)].remove(monitor)

            # if there's no valid monitoring data remaining, discard this event
            if len(self.events_with_monitoring_data[(ring_ID, event, event_type)]) == 0:
                del self.events_with_monitoring_data[(ring_ID, event, event_type)]

        print len(self.events_with_monitoring_data), 'valid events after demand check'
        self.print_monitoring_data_summary('test_demand_variation()')
        # print len(demand_percentile_diff)
        # fig = plt.figure(figsize=(14, 8), facecolor='w')
        # plt.hist(demand_percentile_diff, bins=30)
        # plt.show()

        unique_event_locations_shelf = shelve.open('unique_event_locations')
        unique_event_locations = {}
        for ring_ID, event, event_type in self.events_with_monitoring_data:
            monitors = self.events_with_monitoring_data[(ring_ID, event, event_type)]
            for monitor in monitors:
                location = (ring_ID, monitor['primary_name'], monitor['monitor_name'])
                # print location
                if location not in unique_event_locations:
                    unique_event_locations[location] = 1
                else:
                    unique_event_locations[location] += 1
        print 'len:', len(unique_event_locations)
        for i in unique_event_locations:
            print i, unique_event_locations[i]
        unique_event_locations_shelf['locations'] = unique_event_locations
        unique_event_locations_shelf.close()



    def compare_phase_metrics(self, values_open, values_closed, comparison_type='median', percentile=None):
        if percentile is None:
            percentile = self.PERCENTILE
        lower_closed = {}
        lower_open = {}
        same = {}
        metric_keys = ['Demand', 'Voltage', 'THD', 'TDD', 'Pst', 'Plt']
        for k in metric_keys:
            lower_closed[k] = 0
            lower_open[k] = 0
            same[k] = 0

        for phase_open, phase_closed in zip(values_open, values_closed):
            for k, v, in phase_open.iteritems():
                if comparison_type == 'mean':
                    comparitor_closed = np.mean(phase_closed[k])
                    comparitor_open = np.mean(phase_open[k])
                elif comparison_type == 'percentile':
                    comparitor_closed = np.percentile(phase_closed[k], percentile)
                    comparitor_open = np.percentile(phase_open[k], percentile)
                elif comparison_type == 'median':
                    comparitor_closed = np.median(phase_closed[k])
                    comparitor_open = np.median(phase_open[k])

                if comparitor_closed < comparitor_open:
                    lower_closed[k] += 1
                elif comparitor_closed > comparitor_open:
                    lower_open[k] += 1
                else:
                    same[k] += 1

        print 'comparing', comparison_type + ':'
        for k in metric_keys:
            print ' ', k, 'lower open:', lower_open[k], 'lower closed:', lower_closed[k], 'same:', same[k]

    def compare_all_metrics(self):
        values_closed = []
        values_open = []

        for ring_ID, event, event_type in self.events_with_monitoring_data:
            monitors = self.events_with_monitoring_data[(ring_ID, event, event_type)]
            for monitor in monitors:
                table = monitor['table']
                event_with_offset = event + self.get_offset(monitor['monitor_name'], event)
                start_datetime = event_with_offset - datetime.timedelta(days=7)
                end_datetime = event_with_offset + datetime.timedelta(days=7)

                for i, metric_phase in enumerate(self.metrics):
                    values_before = {}
                    values_after = {}
                    for k, v in metric_phase.iteritems():
                        values_before[k] = [row[v] for row in table.where('(date >= ' + str(calendar.timegm(start_datetime.timetuple())) + ') & (date <= ' + str(calendar.timegm(event_with_offset.timetuple())) + ')')]
                        values_after[k] = [row[v] for row in table.where('(date > ' + str(calendar.timegm(event_with_offset.timetuple())) + ') & (date <= ' + str(calendar.timegm(end_datetime.timetuple())) + ')')]
                    
                    values_closed.append(values_after if event_type == 'Closed' else values_before)
                    values_open.append(values_before if event_type == 'Closed' else values_after)

        self.compare_phase_metrics(values_open, values_closed, comparison_type='mean')
        self.compare_phase_metrics(values_open, values_closed, comparison_type='median')
        self.compare_phase_metrics(values_open, values_closed, comparison_type='percentile')

    def compare_all_metrics_csv(self):
        metric_keys = ['Demand', 'Voltage', 'THD', 'TDD', 'Pst', 'Plt']
        methods = ['Mean', 'Median', '$95^{th}$ percentile']
        output = {k: [] for k in metric_keys}
        data = {k: {method: [[], [], []] for method in methods} for k in metric_keys}
        data_open = {k: {method: [[], [], []] for method in methods} for k in metric_keys}
        data_closed = {k: {method: [[], [], []] for method in methods} for k in metric_keys}
        data_corr = {k: {method: [[], [], []] for method in methods} for k in metric_keys}
        data_corr_interphase = {k: {method: [[], [], []] for method in methods} for k in metric_keys}
        data_corr_interphase_interconn = {k: {method: [[], [], []] for method in methods} for k in metric_keys}

        for ring_ID, event, event_type in self.events_with_monitoring_data:
            monitors = self.events_with_monitoring_data[(ring_ID, event, event_type)]
            for monitor in monitors:
                table = monitor['table']
                event_with_offset = event + self.get_offset(monitor['monitor_name'], event)
                start_datetime = event_with_offset - datetime.timedelta(days=7)
                end_datetime = event_with_offset + datetime.timedelta(days=7)

                values_closed = [[], [], []]
                values_open = [[], [], []]

                for i, metric_phase in enumerate(self.metrics):
                    values_before = {}
                    values_after = {}
                    for k, v in metric_phase.iteritems():
                        values_before[k] = [row[v] for row in table.where('(date >= ' + str(calendar.timegm(start_datetime.timetuple())) + ') & (date <= ' + str(calendar.timegm(event_with_offset.timetuple())) + ')')]
                        values_after[k] = [row[v] for row in table.where('(date > ' + str(calendar.timegm(event_with_offset.timetuple())) + ') & (date <= ' + str(calendar.timegm(end_datetime.timetuple())) + ')')]
                    
                    values_closed[i] = values_after if event_type == 'Closed' else values_before
                    values_open[i] = values_before if event_type == 'Closed' else values_after

                for k in metric_keys:
                    output_line = self.ring_name_from_ID(ring_ID) + ',' + monitor['monitor_name'].title() + ',' + str(event_with_offset) + ',' + event_type + ','
                    for i in range(0, 3):
                        diff = np.mean(values_closed[i][k]) - np.mean(values_open[i][k])
                        # if np.abs(diff) > 60.0:
                        #     print 'diff high:', self.ring_name_from_ID(monitor['ring_ID']), monitor['monitor_name'].title(), event_type
                        #     print '  diff:', diff, 'closed:', np.mean(values_closed[i][k]), 'open:', np.mean(values_open[i][k])
                        #     print ' ', len(monitors)
                        data[k][methods[0]][i].append(diff)
                        data_open[k][methods[0]][i].append(np.mean(values_open[i][k]))
                        data_closed[k][methods[0]][i].append(np.mean(values_closed[i][k]))
                        data_corr[k][methods[0]][i].append(self.corr(values_open[i][k], values_closed[i][k]))
                        data_corr_interphase[k][methods[0]][i].append(self.corr(values_open[i][k], values_open[(i + 1) % 3][k]))
                        data_corr_interphase_interconn[k][methods[0]][i].append(self.corr(values_closed[i][k], values_closed[(i + 1) % 3][k]))
                        output_line += '{:.2f},'.format(diff)
                    for i in range(0, 3):
                        diff = np.median(values_closed[i][k]) - np.median(values_open[i][k])
                        data[k][methods[1]][i].append(diff)
                        data_open[k][methods[1]][i].append(np.mean(values_open[i][k]))
                        data_closed[k][methods[1]][i].append(np.mean(values_closed[i][k]))
                        data_corr[k][methods[1]][i].append(self.corr(values_open[i][k], values_closed[i][k]))
                        data_corr_interphase[k][methods[1]][i].append(self.corr(values_open[i][k], values_open[(i + 1) % 3][k]))
                        data_corr_interphase_interconn[k][methods[1]][i].append(self.corr(values_closed[i][k], values_closed[(i + 1) % 3][k]))
                        output_line += '{:.2f},'.format(diff)
                    for i in range(0, 3):
                        diff = np.percentile(values_closed[i][k], self.PERCENTILE) - np.percentile(values_open[i][k], self.PERCENTILE)
                        data[k][methods[2]][i].append(diff)
                        data_open[k][methods[2]][i].append(np.mean(values_open[i][k]))
                        data_closed[k][methods[2]][i].append(np.mean(values_closed[i][k]))
                        data_corr[k][methods[2]][i].append(self.corr(values_open[i][k], values_closed[i][k]))
                        data_corr_interphase[k][methods[2]][i].append(self.corr(values_open[i][k], values_open[(i + 1) % 3][k]))
                        data_corr_interphase_interconn[k][methods[2]][i].append(self.corr(values_closed[i][k], values_closed[(i + 1) % 3][k]))
                        output_line += '{:.2f},'.format(diff)
                    output_line += '\n'
                    output[k].append(output_line)

        # write a CSV file for each measurement type
        for k in metric_keys:
            with open('csv_data\\PQ_diff_' + k + '.csv', 'wb') as f:
                f.write('Primary,Secondary,Event time,Event type,Phase A mean,Phase B mean,Phase C mean,Phase A median,Phase B median,Phase C median,Phase A 95th percentile,Phase B 95th percentile,Phase C 95th percentile,\n')
                for line in output[k]:
                    f.write(line)

        # per-phase histograms
        colours = ['r', 'y', 'b']
        phase_names = ['A', 'B', 'C']
        max_x_limits = {k: 0.0 for k in metric_keys}
        max_y_limits = {k: 0.0 for k in metric_keys}
        fig, axes = plt.subplots(6, 3, figsize=(14, 20), facecolor='w')
        for i, k in enumerate(metric_keys):
            for m, method in enumerate(methods):
                for p, c, phase_name in zip(range(0, 3), colours, phase_names):
                    axes[i, m].hist(data[k][method][p], normed=False, histtype='step', bins=15, label=phase_name, color=c, alpha=0.7)
                    # axes[i, m].plot(data[k][method][p], label=phase_name, color=c, alpha=0.7)
                    if i == 0:
                        axes[i, m].set_title(method)
                    units = ''
                    if k == 'THD' or k == 'TDD':
                        units = ' (%)'
                    elif k == 'Demand':
                        units = ' (A)'
                    elif k == 'Voltage':
                        units = ' (V)'
                    axes[i, m].set_xlabel('$\delta$ ' + k + units)

                    # centre the x-axis around zero
                    limits = axes[i, m].axis()
                    # print 'limits:', limits
                    x_limit = max(np.abs(limits[0]), np.abs(limits[1]))
                    y_limit = max(np.abs(limits[2]), np.abs(limits[3]))
                    if x_limit > max_x_limits[k]:
                        max_x_limits[k] = x_limit
                    if y_limit > max_y_limits[k]:
                        max_y_limits[k] = y_limit
                    # axes[i, m].set_xlim(-1 * x_limit, x_limit)
                    # axes[i, m].set_ylim(0.0, y_limit)
                    axes[i, m].axvline(0.0, color='k', alpha=0.2, linestyle='--')

                if m == 0 and i == 0:
                    axes[0, 0].legend(loc='best')

        # use the same max range for all x-axes, per metric
        for i, k in enumerate(metric_keys):
            for m, method in enumerate(methods):
                 axes[i, m].set_xlim(-1 * max_x_limits[k], max_x_limits[k])
                 axes[i, m].set_ylim(0.0, max_y_limits[k] + 1)

        # shared y axis label
        fig.text(0.015, 0.5, 'Number of occurrences', ha='center', va='center', rotation='vertical', fontsize=16)

        plt.tight_layout()
        plt.subplots_adjust(left=0.05)#, bottom=0.02, right=0.995, top=0.995, wspace=0.20, hspace=0.80)
        plt.savefig('plots\\compare_all_metrics_csv.pdf')
        plt.savefig('plots\\compare_all_metrics_csv.png', dpi=220)
        # plt.show()


        # test for correlation between differences in data
        corr_metric = 'Median'#'Mean'#'$95^{th}$ percentile'
        print 'correlating with (where relevant):', corr_metric
        print 'correlations between Radial vs. Interconnected:'
        for p, phase_name in zip(range(0, 3), phase_names):
            print '  Demand and Voltage:', phase_name, self.corr(data['Demand'][corr_metric][p], data['Voltage'][corr_metric][p])
        print ''
        for p, phase_name in zip(range(0, 3), phase_names):
            print '  Demand and TDD:    ', phase_name, self.corr(data['Demand'][corr_metric][p], data['TDD'][corr_metric][p])
        print ''
        for p, phase_name in zip(range(0, 3), phase_names):
            print '  TDD and THD:       ', phase_name, self.corr(data['TDD'][corr_metric][p], data['THD'][corr_metric][p])
        print ''
        for p, phase_name in zip(range(0, 3), phase_names):
            print '  Demand and THD:    ', phase_name, self.corr(data['Demand'][corr_metric][p], data['THD'][corr_metric][p])
        print ''

        # test for correlation between differences in data
        print 'correlations for Radial:'
        for p, phase_name in zip(range(0, 3), phase_names):
            print '  Demand and Voltage:', phase_name, self.corr(data_open['Demand'][corr_metric][p], data_open['Voltage'][corr_metric][p])
        print ''
        for p, phase_name in zip(range(0, 3), phase_names):
            print '  Demand and TDD:    ', phase_name, self.corr(data_open['Demand'][corr_metric][p], data_open['TDD'][corr_metric][p])
        print ''
        for p, phase_name in zip(range(0, 3), phase_names):
            print '  TDD and THD:       ', phase_name, self.corr(data_open['TDD'][corr_metric][p], data_open['THD'][corr_metric][p])
        print ''
        for p, phase_name in zip(range(0, 3), phase_names):
            print '  Demand and THD:    ', phase_name, self.corr(data_open['Demand'][corr_metric][p], data_open['THD'][corr_metric][p])
        print ''
        print 'correlations for Interconnected:'
        for p, phase_name in zip(range(0, 3), phase_names):
            print '  Demand and Voltage:', phase_name, self.corr(data_closed['Demand'][corr_metric][p], data_closed['Voltage'][corr_metric][p])
        print ''
        for p, phase_name in zip(range(0, 3), phase_names):
            print '  Demand and TDD:    ', phase_name, self.corr(data_closed['Demand'][corr_metric][p], data_closed['TDD'][corr_metric][p])
        print ''
        for p, phase_name in zip(range(0, 3), phase_names):
            print '  TDD and THD:       ', phase_name, self.corr(data_closed['TDD'][corr_metric][p], data_closed['THD'][corr_metric][p])
        print ''
        for p, phase_name in zip(range(0, 3), phase_names):
            print '  Demand and THD:    ', phase_name, self.corr(data_closed['Demand'][corr_metric][p], data_closed['THD'][corr_metric][p])
        print ''

        # test for correlation between phases
        print 'correlate means between phases (Radial C2C):'
        for i, k in enumerate(metric_keys):
            print '  ' + k + ' phase A and B:', self.corr(data_open[k][corr_metric][0], data_open[k][corr_metric][1])
            print '  ' + k + ' phase B and C:', self.corr(data_open[k][corr_metric][1], data_open[k][corr_metric][2])
            print '  ' + k + ' phase A and C:', self.corr(data_open[k][corr_metric][0], data_open[k][corr_metric][2])
            print ''
        print 'correlate means between phases (Interconnected C2C):'
        for i, k in enumerate(metric_keys):
            print '  ' + k + ' phase A and B:', self.corr(data_closed[k][corr_metric][0], data_closed[k][corr_metric][1])
            print '  ' + k + ' phase B and C:', self.corr(data_closed[k][corr_metric][1], data_closed[k][corr_metric][2])
            print '  ' + k + ' phase A and C:', self.corr(data_closed[k][corr_metric][0], data_closed[k][corr_metric][2])
            print ''

        # compare correlations for 5-minute values for each event
        print 'means of correlations of Radial vs. Interconnected for each event:'
        for i, k in enumerate(metric_keys):
            print '  ' + k + ' phase A:', np.mean(data_corr[k][corr_metric][0])#, data_corr[k]['Mean'][0]
            print '  ' + k + ' phase B:', np.mean(data_corr[k][corr_metric][1])#, data_corr[k]['Mean'][1]
            print '  ' + k + ' phase C:', np.mean(data_corr[k][corr_metric][2])#, data_corr[k]['Mean'][2]
            print ''

        # as above, but inter-phase correlations (Radial)
        print 'means of inter-phase correlations (Radial):'
        for i, k in enumerate(metric_keys):
            print '  ' + k + ' phase A and B:', np.mean(data_corr_interphase[k][corr_metric][0])#, data_corr[k]['Mean'][0]
            print '  ' + k + ' phase B and C:', np.mean(data_corr_interphase[k][corr_metric][1])#, data_corr[k]['Mean'][1]
            print '  ' + k + ' phase C and A:', np.mean(data_corr_interphase[k][corr_metric][2])#, data_corr[k]['Mean'][2]
            print ''

        # as above, but inter-phase correlations (Interconnected)
        print 'means of inter-phase correlations (Interconnected):'
        for i, k in enumerate(metric_keys):
            print '  ' + k + ' phase A and B:', np.mean(data_corr_interphase_interconn[k][corr_metric][0])#, data_corr[k]['Mean'][0]
            print '  ' + k + ' phase B and C:', np.mean(data_corr_interphase_interconn[k][corr_metric][1])#, data_corr[k]['Mean'][1]
            print '  ' + k + ' phase C and A:', np.mean(data_corr_interphase_interconn[k][corr_metric][2])#, data_corr[k]['Mean'][2]
            print ''




    def compare_all_metrics_csv2(self, has_two_monitors_only=False):
        metric_keys = ['Demand', 'Voltage', 'THD', 'TDD', 'Pst', 'Plt']
        methods = ['Mean', 'Median', '$95^{th}$ percentile']
        output = {k: [] for k in metric_keys}
        data = {k: {method: [[], [], []] for method in methods} for k in metric_keys}
        data_open = {k: {method: [[], [], []] for method in methods} for k in metric_keys}
        data_closed = {k: {method: [[], [], []] for method in methods} for k in metric_keys}
        data_corr = {k: {method: [[], [], []] for method in methods} for k in metric_keys}
        data_corr_interphase = {k: {method: [[], [], []] for method in methods} for k in metric_keys}

        for ring_ID, event, event_type in self.events_with_monitoring_data:
            monitors = self.events_with_monitoring_data[(ring_ID, event, event_type)]

            if has_two_monitors_only and len(monitors) != 2:
                continue
            # print 'valid has_two_monitors_only,', len(monitors)

            for monitor in monitors:
                table = monitor['table']
                event_with_offset = event + self.get_offset(monitor['monitor_name'], event)
                start_datetime = event_with_offset - datetime.timedelta(days=7)
                end_datetime = event_with_offset + datetime.timedelta(days=7)

                values_closed = [[], [], []]
                values_open = [[], [], []]

                for i, metric_phase in enumerate(self.metrics):
                    values_before = {}
                    values_after = {}
                    for k, v in metric_phase.iteritems():
                        values_before[k] = [row[v] for row in table.where('(date >= ' + str(calendar.timegm(start_datetime.timetuple())) + ') & (date <= ' + str(calendar.timegm(event_with_offset.timetuple())) + ')')]
                        values_after[k] = [row[v] for row in table.where('(date > ' + str(calendar.timegm(event_with_offset.timetuple())) + ') & (date <= ' + str(calendar.timegm(end_datetime.timetuple())) + ')')]
                    
                    values_closed[i] = values_after if event_type == 'Closed' else values_before
                    values_open[i] = values_before if event_type == 'Closed' else values_after

                for k in metric_keys:
                    output_line = self.ring_name_from_ID(ring_ID) + ',' + monitor['monitor_name'].title() + ',' + str(event_with_offset) + ',' + event_type + ','
                    for i in range(0, 3):
                        diff = np.mean(values_closed[i][k]) - np.mean(values_open[i][k])
                        # if np.abs(diff) > 60.0:
                        #     print 'diff high:', self.ring_name_from_ID(monitor['ring_ID']), monitor['monitor_name'].title(), event_type
                        #     print '  diff:', diff, 'closed:', np.mean(values_closed[i][k]), 'open:', np.mean(values_open[i][k])
                        #     print ' ', len(monitors)
                        data[k][methods[0]][i].append(diff)
                        data_open[k][methods[0]][i].append(np.mean(values_open[i][k]))
                        data_closed[k][methods[0]][i].append(np.mean(values_closed[i][k]))
                        data_corr[k][methods[0]][i].append(self.corr(values_open[i][k], values_closed[i][k]))
                        data_corr_interphase[k][methods[0]][i].append(self.corr(values_open[i][k], values_open[(i + 1) % 3][k]))
                        output_line += '{:.2f},'.format(diff)
                    for i in range(0, 3):
                        diff = np.median(values_closed[i][k]) - np.median(values_open[i][k])
                        data[k][methods[1]][i].append(diff)
                        data_open[k][methods[1]][i].append(np.mean(values_open[i][k]))
                        data_closed[k][methods[1]][i].append(np.mean(values_closed[i][k]))
                        data_corr[k][methods[1]][i].append(self.corr(values_open[i][k], values_closed[i][k]))
                        data_corr_interphase[k][methods[1]][i].append(self.corr(values_open[i][k], values_open[(i + 1) % 3][k]))
                        output_line += '{:.2f},'.format(diff)
                    for i in range(0, 3):
                        diff = np.percentile(values_closed[i][k], self.PERCENTILE) - np.percentile(values_open[i][k], self.PERCENTILE)
                        data[k][methods[2]][i].append(diff)
                        data_open[k][methods[2]][i].append(np.mean(values_open[i][k]))
                        data_closed[k][methods[2]][i].append(np.mean(values_closed[i][k]))
                        data_corr[k][methods[2]][i].append(self.corr(values_open[i][k], values_closed[i][k]))
                        data_corr_interphase[k][methods[2]][i].append(self.corr(values_open[i][k], values_open[(i + 1) % 3][k]))
                        output_line += '{:.2f},'.format(diff)
                    output_line += '\n'
                    output[k].append(output_line)

        # write a CSV file for each measurement type
        for k in metric_keys:
            with open('csv_data\\PQ_diff_' + k + '.csv', 'wb') as f:
                f.write('Primary,Secondary,Event time,Event type,Phase A mean,Phase B mean,Phase C mean,Phase A median,Phase B median,Phase C median,Phase A 95th percentile,Phase B 95th percentile,Phase C 95th percentile,\n')
                for line in output[k]:
                    f.write(line)

        # per-phase histograms
        colours = ['r', 'y', 'b']
        phase_names = ['A', 'B', 'C']
        max_x_limits = {k: 0.0 for k in metric_keys}
        max_y_limits = {k: 0.0 for k in metric_keys}
        fig, axes = plt.subplots(6, 1, figsize=(13, 20), facecolor='w')
        for i, k in enumerate(metric_keys):
            # for m, method in enumerate(methods):
            m = 0
            method = 'Mean'
            # for p, c, phase_name in zip(range(0, 3), colours, phase_names):
            combined_phase_data = data[k][method][0] + data[k][method][1] + data[k][method][2]
            axes[i].hist(combined_phase_data, normed=False, histtype='stepfilled', bins=15, label='All phases', color='b', linewidth=0, alpha=0.8)
            # axes[i, m].plot(data[k][method][p], label=phase_name, color=c, alpha=0.7)
            # if i == 0:
            #     axes[i].set_title(method)
            units = ''
            if k == 'THD' or k == 'TDD':
                units = ' (%)'
            elif k == 'Demand':
                units = ' (A)'
            elif k == 'Voltage':
                units = ' (V)'
            axes[i].set_xlabel('$\delta$ ' + k + units, fontsize=18)

            # centre the x-axis around zero
            limits = axes[i].axis()
            # print 'limits:', limits
            x_limit = max(np.abs(limits[0]), np.abs(limits[1]))
            y_limit = max(np.abs(limits[2]), np.abs(limits[3]))
            if x_limit > max_x_limits[k]:
                max_x_limits[k] = x_limit
            if y_limit > max_y_limits[k]:
                max_y_limits[k] = y_limit
            # axes[i, m].set_xlim(-1 * x_limit, x_limit)
            # axes[i, m].set_ylim(0.0, y_limit)
            axes[i].axvline(0.0, color='k', alpha=0.2, linestyle='--')

            # if m == 0 and i == 0:
            #     axes[0].legend(loc='best')

        # use the same max range for all x-axes, per metric
        for i, k in enumerate(metric_keys):
            # for m, method in enumerate(methods):
            m = 0
            method = 'Mean'
            axes[i].set_xlim(-1 * max_x_limits[k], max_x_limits[k])
            axes[i].set_ylim(0.0, max_y_limits[k] + 1)
            plt.setp(axes[i].get_xticklabels(), fontsize=16)
            plt.setp(axes[i].get_yticklabels(), fontsize=16)

        # shared y axis label
        fig.text(0.015, 0.5, 'Number of occurrences', ha='center', va='center', rotation='vertical', fontsize=18)

        plt.tight_layout()
        plt.subplots_adjust(left=0.06, right=0.978)#, bottom=0.02, right=0.995, top=0.995, wspace=0.20, hspace=0.80)
        plt.savefig('plots\\compare_all_metrics_csv2.pdf')
        plt.savefig('plots\\compare_all_metrics_csv2.png', dpi=220)
        # plt.show()

    def compare_all_metrics_csv2_for_paper(self, has_two_monitors_only=False):
        metric_keys = ['Demand', 'Voltage', 'THD', 'TDD', 'Pst', 'Plt']
        methods = ['Mean', 'Median', '$95^{th}$ percentile']
        output = {k: [] for k in metric_keys}
        data = {k: {method: [[], [], []] for method in methods} for k in metric_keys}
        data_open = {k: {method: [[], [], []] for method in methods} for k in metric_keys}
        data_closed = {k: {method: [[], [], []] for method in methods} for k in metric_keys}
        data_corr = {k: {method: [[], [], []] for method in methods} for k in metric_keys}
        data_corr_interphase = {k: {method: [[], [], []] for method in methods} for k in metric_keys}

        for ring_ID, event, event_type in self.events_with_monitoring_data:
            monitors = self.events_with_monitoring_data[(ring_ID, event, event_type)]

            if has_two_monitors_only and len(monitors) != 2:
                continue
            # print 'valid has_two_monitors_only,', len(monitors)

            for monitor in monitors:
                table = monitor['table']
                event_with_offset = event + self.get_offset(monitor['monitor_name'], event)
                start_datetime = event_with_offset - datetime.timedelta(days=7)
                end_datetime = event_with_offset + datetime.timedelta(days=7)

                values_closed = [[], [], []]
                values_open = [[], [], []]

                for i, metric_phase in enumerate(self.metrics):
                    values_before = {}
                    values_after = {}
                    for k, v in metric_phase.iteritems():
                        values_before[k] = [row[v] for row in table.where('(date >= ' + str(calendar.timegm(start_datetime.timetuple())) + ') & (date <= ' + str(calendar.timegm(event_with_offset.timetuple())) + ')')]
                        values_after[k] = [row[v] for row in table.where('(date > ' + str(calendar.timegm(event_with_offset.timetuple())) + ') & (date <= ' + str(calendar.timegm(end_datetime.timetuple())) + ')')]
                    
                    values_closed[i] = values_after if event_type == 'Closed' else values_before
                    values_open[i] = values_before if event_type == 'Closed' else values_after

                for k in metric_keys:
                    output_line = self.ring_name_from_ID(ring_ID) + ',' + monitor['monitor_name'].title() + ',' + str(event_with_offset) + ',' + event_type + ','
                    for i in range(0, 3):
                        diff = np.mean(values_closed[i][k]) - np.mean(values_open[i][k])
                        # if np.abs(diff) > 60.0:
                        #     print 'diff high:', self.ring_name_from_ID(monitor['ring_ID']), monitor['monitor_name'].title(), event_type
                        #     print '  diff:', diff, 'closed:', np.mean(values_closed[i][k]), 'open:', np.mean(values_open[i][k])
                        #     print ' ', len(monitors)
                        data[k][methods[0]][i].append(diff)
                        data_open[k][methods[0]][i].append(np.mean(values_open[i][k]))
                        data_closed[k][methods[0]][i].append(np.mean(values_closed[i][k]))
                        data_corr[k][methods[0]][i].append(self.corr(values_open[i][k], values_closed[i][k]))
                        data_corr_interphase[k][methods[0]][i].append(self.corr(values_open[i][k], values_open[(i + 1) % 3][k]))
                        output_line += '{:.2f},'.format(diff)
                    for i in range(0, 3):
                        diff = np.median(values_closed[i][k]) - np.median(values_open[i][k])
                        data[k][methods[1]][i].append(diff)
                        data_open[k][methods[1]][i].append(np.mean(values_open[i][k]))
                        data_closed[k][methods[1]][i].append(np.mean(values_closed[i][k]))
                        data_corr[k][methods[1]][i].append(self.corr(values_open[i][k], values_closed[i][k]))
                        data_corr_interphase[k][methods[1]][i].append(self.corr(values_open[i][k], values_open[(i + 1) % 3][k]))
                        output_line += '{:.2f},'.format(diff)
                    for i in range(0, 3):
                        diff = np.percentile(values_closed[i][k], self.PERCENTILE) - np.percentile(values_open[i][k], self.PERCENTILE)
                        data[k][methods[2]][i].append(diff)
                        data_open[k][methods[2]][i].append(np.mean(values_open[i][k]))
                        data_closed[k][methods[2]][i].append(np.mean(values_closed[i][k]))
                        data_corr[k][methods[2]][i].append(self.corr(values_open[i][k], values_closed[i][k]))
                        data_corr_interphase[k][methods[2]][i].append(self.corr(values_open[i][k], values_open[(i + 1) % 3][k]))
                        output_line += '{:.2f},'.format(diff)
                    output_line += '\n'
                    output[k].append(output_line)

        # write a CSV file for each measurement type
        for k in metric_keys:
            with open('csv_data\\PQ_diff_' + k + '.csv', 'wb') as f:
                f.write('Primary,Secondary,Event time,Event type,Phase A mean,Phase B mean,Phase C mean,Phase A median,Phase B median,Phase C median,Phase A 95th percentile,Phase B 95th percentile,Phase C 95th percentile,\n')
                for line in output[k]:
                    f.write(line)

        # per-phase histograms
        colours = ['r', 'y', 'b']
        phase_names = ['A', 'B', 'C']
        max_x_limits = {k: 0.0 for k in metric_keys}
        max_y_limits = {k: 0.0 for k in metric_keys}
        fig, axes = plt.subplots(6, 1, figsize=(13, 15), facecolor='w')
        for i, k in enumerate(metric_keys):
            # for m, method in enumerate(methods):
            m = 0
            method = 'Mean'
            # for p, c, phase_name in zip(range(0, 3), colours, phase_names):
            combined_phase_data = data[k][method][0] + data[k][method][1] + data[k][method][2]
            axes[i].hist(combined_phase_data, normed=False, histtype='stepfilled', bins=15, label='All phases', color='b', linewidth=0, alpha=0.7)
            # axes[i, m].plot(data[k][method][p], label=phase_name, color=c, alpha=0.7)
            # if i == 0:
            #     axes[i].set_title(method)
            units = ''
            if k == 'THD' or k == 'TDD':
                units = ' (%)'
            elif k == 'Demand':
                units = ' (A)'
            elif k == 'Voltage':
                units = ' (V)'
            axes[i].set_xlabel('$\delta$ ' + k + units, fontsize=20)

            # centre the x-axis around zero
            limits = axes[i].axis()
            # print 'limits:', limits
            x_limit = max(np.abs(limits[0]), np.abs(limits[1]))
            y_limit = max(np.abs(limits[2]), np.abs(limits[3]))
            if x_limit > max_x_limits[k]:
                max_x_limits[k] = x_limit
            if y_limit > max_y_limits[k]:
                max_y_limits[k] = y_limit
            # axes[i, m].set_xlim(-1 * x_limit, x_limit)
            # axes[i, m].set_ylim(0.0, y_limit)
            axes[i].axvline(0.0, color='k', alpha=0.2, linestyle='--')

            # if m == 0 and i == 0:
            #     axes[0].legend(loc='best')


        # use the same max range for all x-axes, per metric
        for i, k in enumerate(metric_keys):
            # for m, method in enumerate(methods):
            m = 0
            method = 'Mean'
            # axes[i].tick_params(axis='y', pad=20)
            axes[i].locator_params(axis='y', nbins=6)
            axes[i].set_xlim(-1 * max_x_limits[k] * 1.2, max_x_limits[k] * 1.2)
            axes[i].set_ylim(0.0, max_y_limits[k] + 1)
            plt.setp(axes[i].get_xticklabels(), fontsize=18)
            plt.setp(axes[i].get_yticklabels(), fontsize=18)

        # shared y axis label
        fig.text(0.015, 0.5, 'Number of occurrences', ha='center', va='center', rotation='vertical', fontsize=20)

        plt.tight_layout()
        plt.subplots_adjust(left=0.06, right=0.978)#, bottom=0.02, right=0.995, top=0.995, wspace=0.20, hspace=0.80)
        # plt.savefig('PQ-parameters-comparison.pdf')
        # plt.savefig('PQ-parameters-comparison.png', dpi=220)
        plt.savefig('plots\\PQ-parameters-comparison.eps', format='eps', dpi=1000)
        # plt.show()


    def neg_seq_hist(self, use_percentile=False, plot_mean=False):
        phase_closed = [[], [], []]
        phase_open = [[], [], []]
        bins = 300

        metric = ['IEC_Negative_Sequence_A_Min_perc', 'IEC_Negative_Sequence_A_Avg_perc', 'IEC_Negative_Sequence_A_Max_perc']

        for ring_ID, event, event_type in self.events_with_monitoring_data:
            monitors = self.events_with_monitoring_data[(ring_ID, event, event_type)]
            for monitor in monitors:
                table = monitor['table']
                event_with_offset = event + self.get_offset(monitor['monitor_name'], event)
                start_datetime = event_with_offset - datetime.timedelta(days=7)
                end_datetime = event_with_offset + datetime.timedelta(days=7)

                # values_before = [(row[metric[0]], row[metric[1]], row[metric[2]]) for row in table.where('(date >= ' + str(calendar.timegm(start_datetime.timetuple())) + ') & (date <= ' + str(calendar.timegm(event_with_offset.timetuple())) + ')')]
                values_before = [(row[metric[0]], row[metric[1]], row[metric[2]]) for row in table]
                values_open = values_before
                
                for i in range(0, 3):
                    if use_percentile:
                        phase_open[i].append(np.percentile([v[i] for v in values_open], self.PERCENTILE))
                    elif plot_mean:
                        phase_open[i].append(np.mean([v[i] for v in values_open]))
                    else:
                        phase_open[i].extend([v[i] for v in values_open])

        print '      ', metric
        print 'mean  ', [np.mean(phase_open[0]), np.mean(phase_open[1]), np.mean(phase_open[2])]
        print 'median', [np.median(phase_open[0]), np.median(phase_open[1]), np.median(phase_open[2])]

        colours = ['r', 'y', 'b']
        phase_names = ['Min', 'Average', 'Max']

        fig = plt.figure(figsize=(18, 8), facecolor='w')
        ax1 = plt.subplot(1, 1, 1)
        for phase, label, c, phase_name in zip(phase_open, metric, colours, phase_names):
            ax1.hist(phase, normed=False, histtype='step', bins=bins, label=phase_name, color=c, alpha=0.6)
        ax1.set_xlabel('IEC Negative Sequence Current (%)', fontsize=18)
        ax1.set_ylabel('Number of occurrences', fontsize=18)
        ax1.xaxis.set_ticks(np.arange(0, 1050, 50))
        ax1.legend(loc='best')
        plt.tight_layout()
        plt.savefig('plots\\compare_metric_' + metric[0] + '.pdf')
        plt.savefig('plots\\compare_metric_' + metric[0] + '.png', dpi=220)
        plt.show()


    def compare_metric(self, metric=['THD_V_L1_Avg_perc', 'THD_V_L2_Avg_perc', 'THD_V_L3_Avg_perc'], xlabel='', use_percentile=False, plot_mean=True):
        phase_closed = [[], [], []]
        phase_open = [[], [], []]
        bins = 20

        for ring_ID, event, event_type in self.events_with_monitoring_data:
            monitors = self.events_with_monitoring_data[(ring_ID, event, event_type)]
            for monitor in monitors:
                table = monitor['table']
                event_with_offset = event + self.get_offset(monitor['monitor_name'], event)
                start_datetime = event_with_offset - datetime.timedelta(days=7)
                end_datetime = event_with_offset + datetime.timedelta(days=7)

                values_before = [(row[metric[0]], row[metric[1]], row[metric[2]]) for row in table.where('(date >= ' + str(calendar.timegm(start_datetime.timetuple())) + ') & (date <= ' + str(calendar.timegm(event_with_offset.timetuple())) + ')')]
                values_after = [(row[metric[0]], row[metric[1]], row[metric[2]]) for row in table.where('(date > ' + str(calendar.timegm(event_with_offset.timetuple())) + ') & (date <= ' + str(calendar.timegm(end_datetime.timetuple())) + ')')]
                values_closed = values_after if event_type == 'Closed' else values_before
                values_open = values_before if event_type == 'Closed' else values_after

                if len(values_closed) == len(values_open):
                    for i in range(0, 3):
                        if use_percentile:
                            phase_closed[i].append(np.percentile([v[i] for v in values_closed], self.PERCENTILE))
                            phase_open[i].append(np.percentile([v[i] for v in values_open], self.PERCENTILE))
                        elif plot_mean:
                            phase_closed[i].append(np.mean([v[i] for v in values_closed]))
                            phase_open[i].append(np.mean([v[i] for v in values_open]))
                        else:
                            phase_closed[i].extend([v[i] for v in values_closed])
                            phase_open[i].extend([v[i] for v in values_open])
                            bins = 25

        print '      ', metric
        print 'open  ', [np.mean(phase_open[0]), np.mean(phase_open[1]), np.mean(phase_open[2])]
        print 'closed', [np.mean(phase_closed[0]), np.mean(phase_closed[1]), np.mean(phase_closed[2])]

        colours = ['r', 'y', 'b']
        phase_names = ['Phase A', 'Phase B', 'Phase C']
        # plt.close('all')
        fig = plt.figure(figsize=(18, 6), facecolor='w')
        ax1 = plt.subplot(1, 2, 1)
        for phase, label, c, phase_name in zip(phase_open, metric, colours, phase_names):
            ax1.hist(phase, normed=False, histtype='step', bins=bins, label=phase_name, color=c, alpha=0.6)
        ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
        for phase, label, c, phase_name in zip(phase_closed, metric, colours, phase_names):
            ax2.hist(phase, normed=False, histtype='step', bins=bins, label=phase_name, color=c, alpha=0.6)
        mean_label = ''
        if plot_mean:
            mean_label = 'mean '
        ax1.set_xlabel('Radial $C_2$C, ' + mean_label + xlabel, fontsize=18)
        ax2.set_xlabel('Interconnected $C_2$C, ' + mean_label + xlabel, fontsize=18)
        # if 'demand' in xlabel:
        #     ax1.set_ylim([0.0, 20.0])
        #     ax2.set_ylim([0.0, 20.0])
        ax1.legend(loc='best')
        ax2.legend(loc='best')
        plt.tight_layout()
        plt.savefig('plots\\compare_metric_' + metric[0] + '.pdf')
        plt.savefig('plots\\compare_metric_' + metric[0] + '.png', dpi=220)
        # plt.show()



    def compare_metric_combine_phases(self, metric=['THD_V_L1_Avg_perc', 'THD_V_L2_Avg_perc', 'THD_V_L3_Avg_perc'], xlabel='', use_percentile=False, plot_mean=True):
        phase_closed = []
        phase_open = []
        bins = 20

        for ring_ID, event, event_type in self.events_with_monitoring_data:
            monitors = self.events_with_monitoring_data[(ring_ID, event, event_type)]
            for monitor in monitors:
                table = monitor['table']
                event_with_offset = event + self.get_offset(monitor['monitor_name'], event)
                start_datetime = event_with_offset - datetime.timedelta(days=7)
                end_datetime = event_with_offset + datetime.timedelta(days=7)

                values_before = [(row[metric[0]], row[metric[1]], row[metric[2]]) for row in table.where('(date >= ' + str(calendar.timegm(start_datetime.timetuple())) + ') & (date <= ' + str(calendar.timegm(event_with_offset.timetuple())) + ')')]
                values_after = [(row[metric[0]], row[metric[1]], row[metric[2]]) for row in table.where('(date > ' + str(calendar.timegm(event_with_offset.timetuple())) + ') & (date <= ' + str(calendar.timegm(end_datetime.timetuple())) + ')')]
                values_closed = values_after if event_type == 'Closed' else values_before
                values_open = values_before if event_type == 'Closed' else values_after

                if len(values_closed) == len(values_open):
                    for i in range(0, 3):
                        if use_percentile:
                            phase_closed.append(np.percentile([v[i] for v in values_closed], self.PERCENTILE))
                            phase_open.append(np.percentile([v[i] for v in values_open], self.PERCENTILE))
                        elif plot_mean:
                            phase_closed.append(np.mean([v[i] for v in values_closed]))
                            phase_open.append(np.mean([v[i] for v in values_open]))
                        else:
                            phase_closed.extend([v[i] for v in values_closed])
                            phase_open.extend([v[i] for v in values_open])
                            bins = 25

        print '      ', metric
        print 'open  ', [np.mean(phase_open)]
        print 'closed', [np.mean(phase_closed)]

        # colours = ['r', 'y', 'b']
        # phase_names = ['Phase A', 'Phase B', 'Phase C']
        # plt.close('all')
        fig = plt.figure(figsize=(18, 6), facecolor='w')
        ax1 = plt.subplot(1, 2, 1)
        # for phase, label, c, phase_name in zip(phase_open, metric, colours, phase_names):
        ax1.hist(phase_open, normed=False, histtype='stepfilled', bins=bins, color='r', alpha=0.8, linewidth=0)

        ax2 = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)
        # for phase, label, c, phase_name in zip(phase_closed, metric, colours, phase_names):
        ax2.hist(phase_closed, normed=False, histtype='stepfilled', bins=bins, color='g', alpha=0.8, linewidth=0)

        plt.ylim(0, max(ax1.get_ylim()[1], ax2.get_ylim()[1]) + 10)

        mean_label = ''
        if plot_mean:
            mean_label = 'mean '
        ax1.set_xlabel('Radial $C_2$C, ' + mean_label + xlabel, fontsize=18)
        ax2.set_xlabel('Interconnected $C_2$C, ' + mean_label + xlabel, fontsize=18)
        # if 'demand' in xlabel:
        #     ax1.set_ylim([0.0, 20.0])
        #     ax2.set_ylim([0.0, 20.0])
        # ax1.legend(loc='best')
        # ax2.legend(loc='best')
        plt.tight_layout()
        plt.savefig('plots\\compare_metric_combine_phases_' + metric[0] + '.pdf')
        plt.savefig('plots\\compare_metric_combine_phases_' + metric[0] + '.png', dpi=220)
        # plt.show()


    def visualise_NOP_state_changes(self):
        # visualise NOP state changes
        datemin = datetime.date(2013, 4, 1)
        datemax = datetime.date(2014, 8, 1)
        fig = plt.figure(figsize=(30, 40), facecolor='w')
        axisNum = 0
        for key in sorted(self.sorted_events):
            axisNum += 1
            ax = plt.subplot(len(self.sorted_events), 1, axisNum)
            x = mdates.date2num([e[0] for e in self.sorted_events[key]])
            y = [1.0 if e[1] == 'Closed' else 0.0 for e in self.sorted_events[key]]
            # plt.step(x, y, where='post', color='b', alpha=0.5)
            for start, end, val in self.custom_filter(y):
                mask = np.zeros_like(y)
                mask[start: end] = 1
                ax.fill_between(x, 0.0, 1.0, where=mask, facecolor='b', alpha=0.7, edgecolor='b')
            ax.format_xdata = mdates.DateFormatter('%d/%m/%Y, %H:%M:%S')
            ax.xaxis.set_major_locator(MonthLocator(range(1,13), bymonthday=1, interval=1))
            ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))
            
            for circuit in self.circuits_with_week_data:
                if key == circuit:
                    for event, event_type in self.circuits_with_week_data[circuit]:
                        arrow_datetime = event
                        if (key, event, event_type) in self.events_with_monitoring_data:
                            if len(self.events_with_monitoring_data[(key, event, event_type)]) == 2:
                                arrow_colour = 'g'
                            else:
                                arrow_colour = 'Orange'
                        else:
                            arrow_colour = 'r'
                        arrow_start_datetime = event - datetime.timedelta(days=3, seconds=60*60*8) if event_type == 'Closed' else event + datetime.timedelta(days=3, seconds=60*60*8)
                        ax.annotate("", xy=(arrow_datetime, 0.5), xycoords='data', xytext=(arrow_start_datetime, 0.5), textcoords='data',
                                arrowprops=dict(frac=0.3, headwidth=7.5, width=3.0, alpha=0.9, shrink=0.10, color=arrow_colour))#dict(arrowstyle="-|>", connectionstyle="arc3", alpha=0.9, color=arrow_colour))

            ax.set_yticks([0., 1.0])
            ax.tick_params(axis='x', which='major', color='#AAAAAA', labelsize=14, zorder=-10)
            ax.tick_params(axis='y', which='major', labelsize=14, length=0, width=0)
            labels = [item.get_text() for item in ax.get_yticklabels()]
            labels[0] = 'Open'
            labels[1] = 'Closed'
            ax.set_yticklabels(labels)
            # ax.set_yticks([])

            ax.set_xlim(datemin, datemax)
            ax.spines['top'].set_color('none')
            # ax.spines["top"].set_visible(False)
            ax.spines['right'].set_color('none')
            ax.xaxis.set_ticks_position('bottom')
            plt.ylim([0.0, 1.1])
            # ax.text(datemin + datetime.timedelta(days=1), 0.5, key, horizontalalignment='left', verticalalignment='center', fontsize=8, color='g' if key in self.circuits_with_week_data else 'k')
            ax.set_ylabel(self.ring_name_from_ID(key), rotation=0, fontsize=16, ha='right', va='center')

        plt.gcf().autofmt_xdate()
        plt.subplots_adjust(left=0.10, bottom=0.02, right=0.995, top=0.995, wspace=0.20, hspace=0.80)

        plt.savefig('plots\\visualise_NOP_state_changes.png', dpi=300)
        plt.savefig('plots\\visualise_NOP_state_changes.pdf')


    def plot_values(self, axis_num, event, monitor, event_type, plot_info):
        event_with_offset = event + self.get_offset(monitor['monitor_name'], event)
        start_datetime = event_with_offset - datetime.timedelta(days=7)
        end_datetime = event_with_offset + datetime.timedelta(days=7)
        table = monitor['table']
        
        # values_before = sorted([(d, monitor[plot_info['index']][d]) for d in monitor[plot_info['index']] if d >= start_datetime and d <= event])
        # values_after = sorted([(d, monitor[plot_info['index']][d]) for d in monitor[plot_info['index']] if d > event and d <= end_datetime])
        values_before = [(datetime.datetime.utcfromtimestamp(row['date']), row[plot_info['index']]) for row in table.where('(date >= ' + str(calendar.timegm(start_datetime.timetuple())) + ') & (date <= ' + str(calendar.timegm(event_with_offset.timetuple())) + ')')]
        values_after = [(datetime.datetime.utcfromtimestamp(row['date']), row[plot_info['index']]) for row in table.where('(date > ' + str(calendar.timegm(event_with_offset.timetuple())) + ') & (date <= ' + str(calendar.timegm(end_datetime.timetuple())) + ')')]

        if len(values_before) == 0 or len(values_after) == 0:
            return
        # elif len(values_before) != 7 * 24 * 12 or len(values_before) != 7 * 24 * 12:
        #     print self.ring_name_from_ID(monitor['ring_ID']), monitor['monitor_name'].title(), event_type, len(values_before), len(values_after)

        if event_type == 'Closed':
            color_before = 'r'
            color_after = 'g'
            label_before = 'Radial'
            label_after = 'Interconnected'
        else:
            color_before = 'g'
            color_after = 'r'
            label_before = 'Interconnected'
            label_after = 'Radial'

        ax = plt.subplot(6, 2, axis_num)
        x = [v[0] for v in values_before]
        y = [v[1] for v in values_before]
        plt.plot(x, y, color=color_before)
        # y_line = np.percentile(y, self.PERCENTILE)
        y_line = np.mean(y)
        plt.axhline(y_line, color=color_before, label=label_before, alpha=0.6, linestyle='--')
        x2 = [v[0] for v in values_after]
        y2 = [v[1] for v in values_after]
        plt.plot(x2, y2, color=color_after)
        # y2_line = np.percentile(y2, self.PERCENTILE)
        y2_line = np.mean(y2)
        plt.axhline(y2_line, color=color_after, label=label_after, alpha=0.6, linestyle='--')

        if axis_num == 1 or axis_num == 2:
            ax.set_title(monitor['monitor_name'].title(), fontsize=10)
        formatted_y_label = plot_info['name']
        if plot_info['unit'] != '':
            formatted_y_label += ' (' + plot_info['unit'] + ')'
        ax.set_ylabel(formatted_y_label, fontsize=8)

        ax.format_xdata = mdates.DateFormatter('%d/%m/%Y, %H:%M:%S')
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%a\n%d %b'))
        # ax.xaxis.set_major_formatter(mdates.DateFormatter(''))
        ax.set_ylim(plot_info['limits'][0], plot_info['limits'][1])
        plt.setp(ax.get_xticklabels(), rotation=0, fontsize=8)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

        plt.legend(loc='upper right', ncol=2, shadow=False, labelspacing=0.2, prop={'size': 8})

        plt.axvline(event_with_offset, color='k', alpha=0.3, linestyle='--')
        # arrow_y_pos = limits[0] + (limits[1] - limits[0]) / 8
        # ax.annotate('NOP ' + event_type.lower(), xy=(event, arrow_y_pos), xycoords='data', xytext=(event - datetime.timedelta(days=2), arrow_y_pos), textcoords='data',
        #                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='k', alpha=0.6), alpha=0.6, fontsize=8)


    def visualise_power_quality_metrics(self, has_two_monitors_only=False):
        # visualise changes in power quality metrics
        for ring_ID, event, event_type in self.events_with_monitoring_data:
            monitors = self.events_with_monitoring_data[(ring_ID, event, event_type)]

            if has_two_monitors_only and len(monitors) != 2:
                continue

            if len(monitors) == 1 or len(monitors) == 2:
                fig = plt.figure(figsize=(15, 12), facecolor='w')
                axis_num = 0

                for p in self.plot_types:
                    axis_num += 1
                    self.plot_values(axis_num, event, monitors[0], event_type, plot_info=p)
                    axis_num += 1
                    if len(monitors) == 2:
                        self.plot_values(axis_num, event, monitors[1], event_type, plot_info=p)
                
                # save multiple figures
                plot_title = self.ring_name_from_ID(ring_ID) + ' primary, NOP ' + event_type.lower() + ' on ' + event.strftime('%d %b %Y, %H:%M:%S')
                fig.text(0.5, 0.99, plot_title, horizontalalignment='center', verticalalignment='top', fontsize=14)
                fig.text(0.25, 0.01, 'Time (days)', ha='center', va='center', fontsize=12)
                if len(monitors) == 2:
                    fig.text(0.75, 0.01, 'Time (days)', ha='center', va='center', fontsize=12)
                else:
                    fig.text(0.75, 0.5, 'Other monitoring\nlocation unavailable', ha='center', va='center', fontsize=12, color='k', alpha=0.6)

                plt.gcf().autofmt_xdate(rotation=0, ha='center')
                plt.subplots_adjust(left=0.05, bottom=0.06, right=0.98, top=0.95, wspace=0.20, hspace=0.20)

                plt.savefig('circuit_plots\\' + self.ring_name_from_ID(ring_ID) + ', NOP ' + event_type.lower() + ' on ' + event.strftime('%d %b %Y, %H.%M.%S') + '.pdf')
                plt.close('all')





monitoring_data_processing = MonitoringDataProcessing()
monitoring_data_processing.run()
