from datetime import datetime
import random
import matplotlib.pyplot as plt
import ilya_matplotlib
from collections import defaultdict
import numpy as np
from multiprocessing import Pool

########################################
# standard pacakges
########################################
import sys
import numpy as np
import pickle
import pandas as pd

########################################
# PyQt widgets
########################################
from PyQt5.QtWidgets import QApplication  # main application
from PyQt5.QtWidgets import QMainWindow   # main window
from PyQt5.QtWidgets import QAbstractItemView, QTableView, QHeaderView, QStyleFactory, QSlider, QTabWidget, QRadioButton, QButtonGroup, QCheckBox
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QSizePolicy, QVBoxLayout, QHBoxLayout  # layout on GUI
from PyQt5.QtWidgets import QLabel, QLineEdit, QPushButton         # objects of GUI
# variables for GUI
from PyQt5.QtGui import QFont, QGuiApplication                      # font
from PyQt5 import QtCore
# for visualizing DataFrames
from PyQt5.QtCore import QAbstractTableModel, Qt, QModelIndex, QSortFilterProxyModel

########################################
# Plotting
########################################
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib as mpl


########################################
# ⦿ Kelly Functions
########################################
def kelly_criterion_multiple(bets, probabilities, tax_rate):
    """
    __ Parameters __
    [1D-float] bets:                    for each option
    [1D-float] probabilities:           for each option
    [float] tax_rate:                   between 0 and 1

    __ Description __
    Evaluate fraction of assets to bets on each option

    __ Return __
    [1D-float] assets to bets on each horse
    """

    no_options = len(bets)
    total_bets = sum(bets)

    # 1 - evaluate revenue rate for each option
    revenue_rate = []
    original_index = [i for i in range(0, no_options)]
    for b, s in zip(bets, probabilities):
        revenue_rate.append((1 - tax_rate) * s * total_bets / b)

    # 2 - sort the revenue rates
    idx = np.argsort(revenue_rate)[::-1]
    revenue_rate = np.array(revenue_rate)[idx]
    original_index = np.array(original_index)[idx]

    # 3 - for from best revenue rate and include new options if they fit he wikipedia criteria
    gambling_set = []
    magical_r = 1
    for f_revenue, f_index in zip(revenue_rate, original_index):
        if (f_revenue > magical_r):
            gambling_set.append(f_index)
            numerator = 1 - sum([probabilities[i] for i in gambling_set])
            denominator = 1 - sum([(bets[i] / total_bets)
                                   for i in gambling_set]) / (1 - tax_rate)
            magical_r = numerator / denominator

    # 4 - evaluate the fraction to bets on options that made it to the gambling set
    betting_fraction = []
    for i in gambling_set:
        betting_fraction.append(
            probabilities[i] - magical_r / total_bets * (1 - tax_rate) / bets[i])

    # 5 - return fraction to bets in the order of the orignal options
    betting_fraction_ordered = [0] * no_options
    for f_gambling_set, f_betting_fraction in zip(gambling_set, betting_fraction):
        betting_fraction_ordered[f_gambling_set] = f_betting_fraction

    return betting_fraction_ordered


def multiple_round(balance, fraction_to_bet, bets, probabilities, taxrate):
    """
    __ Parameters __
    [1D-float] balance:                 history of balances
    [1D-float] fraction_to_bet:         on each option in current round
    [1D-float] bets:                    placed on each option
    [1D-float] probabilities:           placed on each option
    [1D-float] taxrate:                 on each bet

    __ Description __
    Play a round of a binary game which amounts to tossing a coin

    __ Return __
    [float] updated balance
    """

    # 1 - place a bet
    temp_balance = balance[-1]

    # 2 - generate random result and find what probability 'bin' it is in
    random_toss = random.random()
    cumulative_probability = 0
    for outcome, i in enumerate(probabilities):
        cumulative_probability += i
        if(cumulative_probability > random_toss):
            break

    # 3 - determine payout given this outcome
    total_bets = sum(bets)
    for idx, bet in enumerate(bets):
        if(idx == outcome):
            temp_balance += temp_balance * \
                fraction_to_bet[idx] * (1 - taxrate) * total_bets / bet
        else:
            temp_balance -= temp_balance * fraction_to_bet[idx]

    balance.append(temp_balance)


def multiple_game(parameters):
    """
    __ Parameters __
    [dict] {'simulation_bets':          [2D-float] during each round for each option
            'simulation_probabilities': [2D-float] during each round for each option
            'simulation_balance':       [1D-float] of balance for each round
            'simulation_taxrate:        [float] on the winnings
            'betting_strategy:          [func] to bet with}
    """

    # 1 - load parameters
    simulation_bets = parameters['simulation_bets']
    simulation_probabilities = parameters['simulation_probabilities']
    simulation_taxrate = parameters['simulation_taxrate']
    simulation_balance = parameters['simulation_balance']
    betting_strategy = parameters['betting_strategy']
    no_rounds = len(simulation_probabilities)

    for i in range(0, no_rounds - 1):

        # 2 - evaluate fraction to bet
        fraction_to_bet = betting_strategy(simulation_bets[i],
                                           simulation_probabilities[i],
                                           simulation_taxrate)

        # 3 - run binary test and with the chosen fraction
        multiple_round(simulation_balance, fraction_to_bet,
                       simulation_bets[i],
                       simulation_probabilities[i],
                       simulation_taxrate)


def kelly_criterion(payout, success=0.5):
    """
    __ Parameters __
    [float] payout:           winning payout on a £1 bet (after £1 has already been covered)
    [float] success:          probability of sucess
    """
    return (success * (payout + 1) - 1) / payout


def binary_round(balance, fraction_to_bet, payout, success):
    """
    __ Parameters __
    [1D-float] balance:         history of balances
    [float] fraction_to_bet:    on in the current round
    [float] payout:             payout on the given bet

    __ Description __
    Play a round of a binary game which amounts to tossing a coin

    __ Return __
    [float] updated balance
    """

    # 1 - place a bet
    temp_balance = balance[-1]

    # 2 - throw coin and depending on result add or remove money
    if(random.random() > success):
        temp_balance -= temp_balance * fraction_to_bet
    else:
        temp_balance += temp_balance * payout * fraction_to_bet

    balance.append(temp_balance)


def binary_game(pool_parameters):
    """
    __ Parameters __
    [dict] pool_parameters:
                                {'simulation_payout':   [1D-float] during each round
                                 'simulation_success':  [1D-float] during each round
                                 'simulation_balance':  [1D-float] of balance for each round
                                 'betting_strategy':    [func] to use for the game}

    __ Description __
    Run a game simulation using the supplied parameters
    """

    # 1 - load parameters
    simulation_payout = pool_parameters['simulation_payout']
    simulation_success = pool_parameters['simulation_success']
    simulation_balance = pool_parameters['simulation_balance']
    betting_strategy = pool_parameters['betting_strategy']
    no_rounds = len(simulation_payout)

    for i in range(0, no_rounds - 1):

        # 2 - evaluate fraction to bet
        fraction_to_bet = betting_strategy(
            simulation_payout[i], simulation_success[i])

        # 3 - run binary test and with the chosen fraction
        binary_round(simulation_balance, fraction_to_bet,
                     simulation_payout[i], simulation_success[i])


class App(QMainWindow):
    """Class that combines all of the widgets"""

    def __init__(self, binary_parameters, multiple_parameters,
                 initial_balance=100, no_tests=1000, no_games_per_test=100):
        """
        __ Parameters __
        [dict] binary_parameters:   {'constant_probability':    when rules of game do not change,
                                     'constant_payout':         when rules of game do not change,
                                     'random_payout_factor':    when payout is scaled randomly,
                                     'betting_functions':       [dict] betting_functions 🍄}
        [dict] betting_functions:   {'🍄': lambda p,s: function(p,s)}

        ------------------------------------------------------------------------------------------

        [dict] multiple_parameters: {'multiple_options':        [int] number of multiple options,
                                     'tax_rate':                [float] between 0 and 1 of deduction from winnings,
                                     'betting_functions':       [dict] betting_functions 🐋}
        [dict] betting_functions:   {'🐋': lambda p, b, t: function(p, b, t)}
        """

        # 1 - Mutltiple inheritance initialization
        super().__init__()

        # 2 - initial values and class costants
        self.constant_bet = 0.2
        self.round_to_display = 10
        self.simulations_to_evaluate = ['Constant Bet', 'Kelly']
        self.multiple_simulations_to_evaluate = [
            'Kelly-Multiple', 'Constant-Multiple']
        self.simulation_choice = 'binary'

        self.initial_balance = initial_balance
        self.no_tests = no_tests
        self.no_games_per_test = no_games_per_test
        self.simulation_global = defaultdict(
            lambda: np.empty((0, self.no_games_per_test)))

        ########################################
        # ⦿ Binary simulation variables
        ########################################
        self.constant_probability = binary_parameters['constant_probability']
        self.constant_payout = binary_parameters['constant_payout']
        self.random_payout_factor = binary_parameters['random_payout_factor']
        self.binary_simulations_colours = [
            'C1', 'C3', 'C6', 'C9', 'C2', 'C4', 'C5']
        self.binary_simulations = {'Constant Bet': lambda p, s: self.constant_bet,
                                   'Kelly': lambda p, s: kelly_criterion(p, s),
                                   'Kelly-Half': lambda p, s: kelly_criterion(p, s) / 2,
                                   'Kelly-Double': lambda p, s: min(kelly_criterion(p, s) * 2, 1),
                                   **binary_parameters['betting_functions']}

        ########################################
        # ⦿ Multiple simulation variables
        ########################################
        self.multiple_options = multiple_parameters['multiple_options']
        self.taxrate = multiple_parameters['tax_rate']
        self.multiple_simulations_colours = [
            'C2', 'C5', 'C1', 'C3', 'C4', 'C6', 'C7', 'C8', 'C9']
        self.multiple_simulations = {'Constant-Multiple':
                                     lambda b, p, t: self.constant_bet *
                                     np.ones(self.multiple_options),
                                     'Kelly-Multiple':
                                     lambda b, p, t: kelly_criterion_multiple(
                                         b, p, t),
                                     **multiple_parameters['betting_functions']}

        # 3 - initilize the look and widgets
        self.init_UI()
        self.read_settings()

        # 4 - run first simulation
        self.simulation(self.simulation_binary)

    def init_UI(self, width=1500, height=600):
        """
        __ Parameters __
        [int] width, height:    of the application

        __ Description __
        set the general look of the gui
        """

        ########################################
        # ⦿ 1 - general parametesr
        ########################################
        self.setWindowTitle("Kelly Criterion")
        self.setGeometry(50, 50, width, height)

        ########################################
        # ⦿ 2 - widgets
        ########################################
        self.widget_histogram = HistogramCanvas(
            self.no_tests, height=8, width=5)
        self.widget_scatter = ScatterCanvas(height=8, width=5)

        ########################################
        # ⦿ 2a - round slider
        ########################################
        self.slider_roundNo = QSlider(Qt.Horizontal)
        self.slider_roundNo.setFocusPolicy(Qt.StrongFocus)
        self.slider_roundNo.setSingleStep(1)
        self.slider_roundNo.setMinimum(0)
        self.slider_roundNo.setMaximum(self.no_games_per_test - 1)
        self.slider_roundNo.setValue(self.round_to_display)
        self.slider_roundNo.sliderMoved.connect(self.simulation_plot)

        self.label_roundNo = QLabel(self)
        self.label_roundNo.setText(f"Round {self.round_to_display}")

        ########################################
        # ⦿ 2b - tabs
        ########################################
        self.tabs = QTabWidget()
        self.tabBinary = QWidget()
        self.tabMultiple = QWidget()
        self.tabs.addTab(self.tabBinary, "Binary")
        self.tabs.addTab(self.tabMultiple, "Multiple")

        ########################################
        # ⦿ 2c - simulation parameters
        ########################################
        self.input_bet = QLineEdit(self)
        self.input_bet.setMaximumWidth(100)
        self.input_bet.setText(str(self.constant_bet))

        self.radio_randomSuccess = QRadioButton(
            f"On - Random Probability (0 - 1)\nOff - Constant Probability ({self.constant_probability})")
        self.radio_randomPayout = QRadioButton(
            f"On - Random Payout (1 - {self.random_payout_factor})\nOff - Consant Payout ({self.constant_payout})")

        ########################################
        # ⦿ 🍄 - radio buttons
        ########################################
        # a - radio buttons are stored in a dictionary
        self.radioButtons_binary = {}

        # b - the binary group allows them all to be turned on
        self.group_binary = QButtonGroup()

        # c - go over all the possible binary simulations and generate radio buttons
        for i in self.binary_simulations.keys():
            self.radioButtons_binary[i] = QRadioButton(i)
            self.group_binary.addButton(self.radioButtons_binary[i])

        self.group_binary.addButton(self.radio_randomSuccess)
        self.group_binary.addButton(self.radio_randomPayout)

        # d - allow them to be turned on in parrallel
        self.group_binary.setExclusive(False)

        # e - turn on default ones
        for i in self.simulations_to_evaluate:
            self.radioButtons_binary[i].setChecked(True)

        ########################################
        # 🐋 - radio buttons for multiple
        ########################################
        # a - radio buttons are stored in a dictionary
        self.radioButtons_multiple = {}

        # b - the binary group allows them all to be turned on
        self.group_multiple = QButtonGroup()

        # c - go over all the possible binary simulations and generate radio buttons
        for i in self.multiple_simulations.keys():
            self.radioButtons_multiple[i] = QRadioButton(i)
            self.group_multiple.addButton(self.radioButtons_multiple[i])

        # d - allow them to be turned on in parrallel
        self.group_multiple.setExclusive(False)

        ########################################
        # ⦿ 2c - buttons to launch simulations
        ########################################
        self.button_binarySimulation = QPushButton(
            f"Binary simulation ({self.no_tests} tests)")

        def button_binary_lambda():
            self.simulation_choice = 'binary'
            return self.simulation(self.simulation_binary)
        self.button_binarySimulation.pressed.connect(button_binary_lambda)

        self.button_multipleSimulation = QPushButton(
            f"Multiple simulation ({self.no_tests} tests)")

        def button_multiple_lambda():
            self.simulation_choice = 'multiple'
            return self.simulation(self.simulation_multiple)
        self.button_multipleSimulation.pressed.connect(button_multiple_lambda)

        ########################################
        # ⦿ 3 - layout
        ########################################
        layout_binary_parameters = QHBoxLayout()
        layout_binary_parameters.addWidget(self.radio_randomSuccess)
        layout_binary_parameters.addWidget(self.radio_randomPayout)

        layout_binary_top = QHBoxLayout()
        layout_binary_top.addWidget(self.radioButtons_binary['Constant Bet'])
        layout_binary_top.addWidget(self.input_bet)

        layout_binary = QVBoxLayout()
        layout_binary.addLayout(layout_binary_parameters)
        layout_binary.addLayout(layout_binary_top)
        for i in self.binary_simulations.keys():
            layout_binary.addWidget(self.radioButtons_binary[i])
        layout_binary.addWidget(self.button_binarySimulation)
        self.tabBinary.setLayout(layout_binary)

        layout_multiple = QVBoxLayout()
        for i in self.multiple_simulations.keys():
            layout_multiple.addWidget(self.radioButtons_multiple[i])
        layout_multiple.addWidget(self.button_multipleSimulation)
        self.tabMultiple.setLayout(layout_multiple)

        layout_round_slider = QHBoxLayout()
        layout_round_slider.addWidget(self.slider_roundNo)
        layout_round_slider.addWidget(self.label_roundNo)

        layout_graphs = QHBoxLayout()
        layout_graphs.addWidget(self.widget_histogram)
        layout_graphs.addWidget(self.widget_scatter)

        layout_1 = QVBoxLayout()
        layout_1.addLayout(layout_round_slider)
        layout_1.addLayout(layout_graphs)
        # layout_1.addLayout(layout_buttons)

        layout_2 = QHBoxLayout()
        layout_2.addLayout(layout_1)
        layout_2.addWidget(self.tabs)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        central_widget.setLayout(layout_2)

        self.show()

    def simulation_multiple(self):
        """
        __ Description __
        Run simulation on the case of multiple outcomes
        """

        for i in range(0, self.no_tests):

            # 2 - bets on the horses for each round (1 to 50)
            self.simulation_bets = np.random.randint(
                1, 50, size=(self.no_games_per_test, self.multiple_options))

            # 3 - probabilities for each option this round
            self.simulation_probabilities = np.random.rand(
                self.no_games_per_test, self.multiple_options)
            for idx, i in enumerate(self.simulation_probabilities):
                self.simulation_probabilities[idx] = self.simulation_probabilities[idx] / sum(
                    i)
            random.seed(datetime.now())
            balance = defaultdict(lambda: [self.initial_balance])

            # 2 - simulation for each of the multiple strategies
            for i in self.multiple_simulations.keys():
                multiple_game({'simulation_bets': self.simulation_bets,
                               'simulation_probabilities': self.simulation_probabilities,
                               'simulation_balance': balance[i],
                               'simulation_taxrate': self.taxrate,
                               'betting_strategy': self.multiple_simulations[i]})

            # 3 - storage of results after each test
            for i in self.multiple_simulations.keys():
                self.simulation_global[i] = np.vstack((self.simulation_global[i],
                                                       np.array(balance[i])))

        # 6 - evaluate average
        for i in self.multiple_simulations.keys():
            self.simulation_global[f'{i}-average'] = self.simulation_global[i].mean(
                axis=0)

    def simulation_binary(self):
        """
        __ Description __
        Runs simulation for the binary game
        """

        for i in range(0, self.no_tests):

            # 1 - generate a random seed for each test and reset the history list
            random.seed(datetime.now())
            balance = defaultdict(lambda: [self.initial_balance])

            # 2 - simulation for each of the binary strategies
            for i in self.binary_simulations.keys():
                binary_game({'simulation_payout': self.simulation_payout,
                             'simulation_success': self.simulation_success,
                             'simulation_balance': balance[i],
                             'betting_strategy': self.binary_simulations[i]})

            # 3 - storage of results after each test
            for i in self.binary_simulations.keys():
                self.simulation_global[i] = np.vstack((self.simulation_global[i],
                                                       np.array(balance[i])))

        # 6 - evaluate average
        for i in self.binary_simulations.keys():
            self.simulation_global[f'{i}-average'] = self.simulation_global[i].mean(
                axis=0)

    def simulation(self, simulation_function):
        """
        __ Parameters __
        [func] simulation_function:  to run

        __ Description __
        Run the specified simulation
        """
        ########################################
        # ⦿ 1 - Parametesr reset
        ########################################
        self.simulation_global = defaultdict(
            lambda: np.empty((0, self.no_games_per_test)))

        ########################################
        # ⦿ 2 - game rules loadded
        ########################################
        self.read_settings()

        ########################################
        # ⦿ 3 - run simulatio
        ########################################
        simulation_function()

        ########################################
        # ⦿ 4 - initial plots
        ########################################
        self.widget_scatter.plot_scatter(self.simulation_global, self.simulations_to_evaluate,
                                         self.simulations_to_evaluate_colours, self.round_to_display)

        self.widget_histogram.plot_histogram(self.simulation_global, self.round_to_display,
                                             self.simulations_to_evaluate,
                                             self.simulations_to_evaluate_colours)
        print("⦿ Finished")

    def simulation_plot(self):
        ########################################
        # ⦿ 1 - update values
        ########################################
        self.read_settings()
        self.round_to_display = self.slider_roundNo.value()
        self.label_roundNo.setText(f"Round {self.round_to_display}")

        ########################################
        # ⦿ 2 - update plot
        ########################################
        self.widget_histogram.plot_histogram(self.simulation_global, self.round_to_display,
                                             self.simulations_to_evaluate,
                                             self.simulations_to_evaluate_colours)
        self.widget_scatter.plot_scatter(self.simulation_global, self.simulations_to_evaluate,
                                         self.simulations_to_evaluate_colours, self.round_to_display)

    def read_settings(self):
        """
        __ Description __
        Reads the state of:
        - radio buttons to generate a list of simulations to perform
        - simulation parameters
        """

        ########################################
        # ⦿ 1 - binary
        ########################################
        if (self.simulation_choice == 'binary'):
            # 1a - simulations to run
            self.simulations_to_evaluate = []
            self.simulations_to_evaluate_colours = []

            for i, colour in zip(self.binary_simulations, self.binary_simulations_colours):
                if self.radioButtons_binary[i].isChecked():
                    self.simulations_to_evaluate.append(i)
                    self.simulations_to_evaluate_colours.append(colour)

            # 1b - constant bet in each round
            self.constant_bet = min(float(self.input_bet.text()), 0.99)
            self.input_bet.setText(str(self.constant_bet))

            # 1c - payout (if random use the number as scale, if constant use it as the payout)
            if(self.radio_randomPayout.isChecked):
                self.simulation_payout = np.random.rand(
                    self.no_games_per_test) * self.random_payout_factor + 1
            else:
                self.simulation_payout = [
                    self.constant_payout] * self.no_games_per_test

            # 1d - probability of success
            if(self.radio_randomSuccess.isChecked):
                self.simulation_success = np.random.rand(
                    self.no_games_per_test)
            else:
                self.simulation_success = [
                    self.constant_probability] * self.no_games_per_test

        ########################################
        # ⦿ 2 - multiple
        ########################################
        else:
            self.simulations_to_evaluate = []
            self.simulations_to_evaluate_colours = []

            for i, colour in zip(self.multiple_simulations, self.multiple_simulations_colours):
                if self.radioButtons_multiple[i].isChecked():
                    self.simulations_to_evaluate.append(i)
                    self.simulations_to_evaluate_colours.append(colour)

            # self.simulations_to_evaluate = self.multiple_simulations.keys()
            # self.simulations_to_evaluate_colours = self.multiple_simulations_colours


class HistogramCanvas(FigureCanvas):
    """class to plot bar charts"""

    def __init__(self, no_tests, parent=None, width=5, height=4, dpi=100):
        """
        [PyQt5] parent: parent object
        [int] width, height, dpi
        """

        # 1 - set class variables
        self.no_tests = no_tests

        # 2 - format axes to plot on
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_axes([0.15, 0.15, 0.8, 0.8])
        self.ax.set_xlabel("Balance")
        self.ax.set_ylabel("Freq")

        # 3 - initialize widget for embededing into the GUI
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot_histogram(self, simulations, round_to_display,
                       simulations_to_evaluate, colours_to_show):
        """
        __ Parameters __
        [dict] simulations:             dictionary of balance histories
        [int] round_to_display:         which round to visualize
        [1D-string] simulations_to_evaluate: list of simulations to show
        [1D] colours_to_show:           for each of the plots

        __ Description __
        Plot the chosen histograms in the chosen colours
        """

        ########################################
        # ⦿ Evaluate range to show
        ########################################
        max_average = 0
        for i in simulations_to_evaluate:
            max_average = max(max_average, max(
                simulations[f'{i}-average'][:round_to_display + 1]))
        self.bins = np.linspace(0, max_average, 100)

        self.ax.clear()
        for idx, i in enumerate(simulations_to_evaluate):

            # 1 - plot histogram
            counts, bins = np.histogram(simulations[i][:, round_to_display],
                                        bins=self.bins)
            self.ax.hist(bins[:-1], bins,
                         weights=counts, alpha=0.5,
                         facecolor=colours_to_show[idx], label=i)

            # 2 - plot average
            # average = simulations[f'{i}-average'][round_to_display]
            # self.ax.axvline(average, linestyle='--',
            # color=colours_to_show[idx])

        # 3 - annotation
        self.ax.legend(shadow=True, ncol=2, loc='upper center')
        self.ax.set_xlabel("Balance")
        self.ax.set_ylabel("Freq")

        # 4 - set range to cover 90% of spread
        # coverage = self.no_tests
        # i = len(counts) - 1
        # while(coverage > 0.9 * self.no_tests):
        #     coverage -= counts[i]
        #     i -= 1
        #     i = max(0, i)
        # self.ax.set_xlim((0, bins[i + 1]))
        self.draw()


class ScatterCanvas(FigureCanvas):
    """class to plot bar charts"""

    def __init__(self, parent=None, width=8, height=4, dpi=100):
        """
        [PyQt5] parent: parent object
        [int] width, height, dpi
        """

        # 1 - format axes to plot on
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_axes([0.15, 0.15, 0.8, 0.8])
        self.ax.set_xlabel("Round")
        self.ax.set_ylabel("Balance")
        self.ax.xaxis.label.set_color('#e7815f')
        self.ax.tick_params(axis='x', colors='#e7815f')
        self.ax.yaxis.label.set_color('#e7815f')
        self.ax.tick_params(axis='y', colors='#e7815f')

        # self.marker_style = dict(linestyle='--', marker='o',
        # markersize=15, markerfacecoloralt='tab:red')

        # 3 - initialize widget for embededing into the GUI
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot_scatter(self, simulations, simulations_to_evaluate, colours_to_show, round_no):
        """
        __ Parameters __
        [dict] simulations:             dictionary of balance histories
        [1D-string] simulations_to_evaluate: list of simulations to show
        [1D] colours_to_show:           for each of the plots
        [int] round_no:                 currently shown

        __ Description __
        Plot the chosen histograms in the chosen colours
        """

        max_y = 0

        self.ax.clear()
        for idx, i in enumerate(simulations_to_evaluate):

            data = simulations[f"{i}-average"]
            # data = simulations[i][0]

            # 1 - plot scatter and moving point
            self.ax.plot(data,
                         color=colours_to_show[idx], label=i)
            self.ax.scatter(
                round_no, data[round_no], color=colours_to_show[idx], linestyle='--')

            # 2 - set the new limit
            max_y = max(max_y, data[round_no] * 1.2)
            # max_y = max(data[int(round_no * 0.8): round_no + 1])
            # max_y = max(max(data[int(round_no * 0.8):round_no + 1]), max_y)

        # 3 - annotation
        self.ax.set_ylim((0, max_y))
        self.ax.set_xlim((round_no * 0.8, round_no * 1.2))
        self.ax.set_xlabel("Round")
        self.ax.set_ylabel("Balance")
        self.draw()


app = QApplication([])


def custom_binary(p, s):
    return p + s


def custom_multiple(b, p, t):
    return [2] * len(b)


# add custom functions with names here
binary_betting_functions = {'🍄': lambda p, s: custom_binary(p, s)}
binary_parameters = {'constant_probability': 0.505,
                     'constant_payout': 2,
                     'random_payout_factor': 2,
                     'betting_functions': binary_betting_functions}

multiple_betting_functions = {'🐳': lambda b, p, t: custom_multiple(b, p, t)}
multiple_parameters = {'multiple_options': 6,
                       'tax_rate': 0.05,
                       'betting_functions': multiple_betting_functions}

gui_box = App(binary_parameters, multiple_parameters,
              no_games_per_test=200, no_tests=100)
sys.exit(app.exec_())
