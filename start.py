# This part is for checking whats missing
try:
    # At the top of your start.py script, after importing necessary modules
    import psycopg2
    import pygame
    import csv
    import pickle
    import math
    from enum import Enum
    import os
    import subprocess
    import importlib
    import chemicals
    import fluids
    import thermo
    import ht
    print("Import Success")
except:
    print("Import Error")
    subprocess.run(["pip", "install", "-r", "requirements.txt"])

current_directory = os.path.dirname(os.path.abspath(__file__))

required_libraries = [
    "subprocess",
    "importlib",
    "pygame",
    "scipy",
    "thermo",
    "fluids",
    "ht",
    "numpy",
    "csv",
    "pickle",
    "math",
    "enum",
    "os",
    "sys",
    "time",
    "random",
    "datetime",
    "tkinter",
    "fpdf",
    "pandas",
    "matplotlib",
    "chemicals",
    "openpyxl",
    #"scikitlearn",
    "pyswarms",
    "pysindy",
    "psycopg2"
    # any other libraries you want to ensure are installed
]

missing_libraries = []

for library in required_libraries:
    try:
        importlib.import_module(library)
        print(f"{library} is installed.")
    except ImportError:
        print(f"{library} is not installed.")
        missing_libraries.append(library)

if missing_libraries:
    print("Attempting to install missing libraries...")
    subprocess.run(["pip", "install", *missing_libraries])
    print("Installation of missing libraries completed.")


# Function to get database connection
def get_db_connection(config_file='chemical_databases/db_config.txt'):
    try:
        # Initialize a dictionary to hold the configuration
        config = {}
        
        # Read configuration from file
        with open(config_file, 'r') as file:
            for line in file:
                key, value = line.strip().split('=')
                config[key] = value

        # Establish the database connection
        return psycopg2.connect(
            dbname=config['dbname'],
            user=config['user'],
            password=config['password'],
            host=config['host']
        )

    except Exception as e:
        print("Error while connecting to the database:", e)
        return None

# Function to check and create the database if it doesn't exist
def setup_database():
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        # Create tables if they don't exist (example for a chemical_data table)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chemical_data (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100),
                density FLOAT,
                viscosity FLOAT,
                ...
            )
        ''')
        conn.commit()
        cursor.close()
        conn.close()
        print("Database setup completed.")
    else:
        print("Database setup failed.")
        print("Please try running setup_db.py")

# Run the database setup function
setup_database()

flowsheet_version = "Flowsheet Simulator v.1.0.0"
# Get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128, 128)
BLUE = (0, 0, 255)

# Initialize Pygame
pygame.init()
info = pygame.display.Info()
screen_width = info.current_w
screen_height = info.current_h

# Set up the Pygame window
scale_factor = 0.8
window_width = int(screen_width * scale_factor)
window_height = int(screen_height * scale_factor)
window_size = (window_width, window_height)
WINDOW_SIZE = (screen_width, screen_height)
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Frank's Chemical Process Simulator")

# Set up fonts
font = pygame.font.Font(None, 40)

# Define button size as a percentage of the window size
button_width_percent = 0.2
button_height_percent = 0.1

# Calculate button size based on the window size
button_width = int(window_width * button_width_percent)
button_height = int(window_height * button_height_percent)


# Define buttons
button_padding = 50
button_start_x = (WINDOW_SIZE[0] - button_width) / 4
button_start_y = (WINDOW_SIZE[1] - (button_height + button_padding) * 5) / 2

# Define the font size as a percentage of the window size
font_size_percent = 0.05

# Calculate the font size based on the window size
font_size = int(window_height * font_size_percent)

# Create a font object
font = pygame.font.SysFont(None, font_size)

# Defines the abstract class called page
class Page:
    """an abstract class representing a single screen"""

    def render(self, pygame_screen):
        pass
# Defines the class called PageManager
class PageManager:
    def __init__(self, initial_page):
        self.current_page = initial_page
        self.current_page.manager = self  # set the manager attribute of the initial page
        self.running = True

    def go_to(self, page):
        self.current_page = page
        self.current_page.manager = self  # set the manager attribute of the new page

    def handle_event(self, event):
        self.current_page.handle_event(event)

    def render(self, screen):
        self.current_page.render(screen)
# Defines the Main Menu pages
class MainMenuPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.in_data_processing = True
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.in_statistics = True
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.in_thermodynamics = True
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        elif self.in_thermodynamics:
            self.manager.go_to(ThermodynamicsPage())
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Data Processing Page
class DataProcessingPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(5)]
        
        # New Buttons
        #self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Matrix Interpolation Filler", "Remove Duplicates", "Temporal Resampling", "Subplotter", "Back"]
        
        # New Button Texts
        #self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_back = False
        self.in_matrix_interpolation_filler = False
        self.in_remove_duplicates = False
        self.in_temporal_resampling = False
        self.in_subplotter = False
        self.in_back = False



        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_matrix_interpolation_filler = True
                    elif i == 1:
                        self.in_remove_duplicates = True
                    elif i == 2:
                        self.in_temporal_resampling = True
                    elif i == 3:
                        self.in_subplotter = True
                    elif i == 4:
                        self.in_back = True
                        self.manager.go_to(MainMenuPage())
        if self.in_matrix_interpolation_filler:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Data_Matrix_Filler_Interpolation.py")])
            self.in_matrix_interpolation_filler = False
        elif self.in_remove_duplicates:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Data_Remove_Duplicates.py")])
            self.in_remove_duplicates = False
        elif self.in_temporal_resampling:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Data_Temporal_Resampling.py")])
            self.in_temporal_resampling = False
        elif self.in_subplotter:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Data_Subplotter.py")])
            self.in_subplotter = False
        elif self.in_back:
            self.manager.go_to(MainMenuPage())
    def exit(self):
        # Terminate the subprocess if it exists before navigating away from this page
        if self.subprocess_obj is not None and self.subprocess_obj.poll() is None:
            self.subprocess_obj.terminate()
  

    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Data Pre Processing", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Statistics Page
class StatisticsPage(Page):
    def __init__(self, page_manager=None):
        self.menu_texts = ["Spearman Pearson Testing",
                            "ANOVA", "Check Consistency",
                            "FFT", "Gaussian Filter",
                            "Matrix Filler Interpolate",
                            "Moving Average Filter",
                            "Random Forest -> Particle Swarm Optimization",
                            "Reduction PCA","PIML","Back"]
        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        column_limit = 5  # Number of buttons per column
        number_of_columns = -(-len(self.menu_texts) // column_limit)  # Ceiling division to get the number of columns needed
        
        self.menu_rects = [
            pygame.Rect(
                button_start_x + (button_width + column_padding) * (i // column_limit),  # X-coordinate: moves to the next column every 'column_limit' buttons
                button_start_y + (button_height + button_padding) * (i % column_limit),  # Y-coordinate: cycles every 'column_limit' buttons
                button_width,
                button_height
            )
            for i in range(len(self.menu_texts))]
        # Existing Button Texts
        self.in_spearman_pearson = False
        self.in_anova = False
        self.in_check_consistency = False
        self.in_fft = False
        self.in_gaussian_filter = False
        self.in_matrix_filler_interpolate = False
        self.in_moving_average_filter = False
        self.in_random_forest = False
        self.in_reduction_pca = False
        self.in_PIML = False
        self.in_back = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_spearman_pearson = True
                    elif i == 1:
                        self.in_anova = True
                    elif i == 2:
                        self.in_check_consistency = True
                    elif i == 3:
                        self.in_fft = True
                    elif i == 4:
                        self.in_gaussian_filter = True
                    elif i == 5:
                        self.in_matrix_filler_interpolate = True
                    elif i == 6:
                        self.in_moving_average_filter = True
                    elif i == 7:
                        self.in_random_forest = True
                    elif i == 8:
                        self.in_reduction_pca = True
                    elif i == 9:
                        self.in_PIML = True
                    elif i == 10:
                        self.in_back = True

        if self.in_spearman_pearson:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Data_Spearman_Pearson_v2.py")])
            self.in_spearman_pearson = False
        elif self.in_anova:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Data_ANOVA.py")])
            self.in_anova = False
        elif self.in_check_consistency:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Data_Check_Consistency.py")])
            self.in_check_consistency = False
        elif self.in_fft:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Data_FFT_Technique.py")])
            self.in_fft = False
        elif self.in_gaussian_filter:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Data_Gaussian_Filter.py")])
            self.in_gaussian_filter = False
        elif self.in_matrix_filler_interpolate:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Data_Matrix_Filler_Interpolation.py")])
            self.in_matrix_filler_interpolate = False
        elif self.in_moving_average_filter:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Data_Moving_Average_Smoothing_Filter.py")])
            self.in_moving_average_filter = False
        elif self.in_random_forest:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Data_Optimizer_RandomForest_ParticleSwarm.py")])
            self.in_random_forest = False
        elif self.in_reduction_pca:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Data_Reduction_PCA_SVD.py")])
            self.in_reduction_pca = False
        elif self.in_PIML:
            self.manager.go_to(PIMLPage())
        elif self.in_back:
            self.manager.go_to(MainMenuPage())
    def exit(self):
        # Terminate the subprocess if it exists before navigating away from this page
        if self.subprocess_obj is not None and self.subprocess_obj.poll() is None:
            self.subprocess_obj.terminate()
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Industrial Statistics", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Physics Informed Machine Learning Page
class PIMLPage(Page):
    def __init__(self, page_manager=None):
        self.menu_texts = ["Genetic Programming","SINDY","Back"]
        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        column_limit = 5  # Number of buttons per column
        number_of_columns = -(-len(self.menu_texts) // column_limit)  # Ceiling division to get the number of columns needed
        
        self.menu_rects = [
            pygame.Rect(
                button_start_x + (button_width + column_padding) * (i // column_limit),  # X-coordinate: moves to the next column every 'column_limit' buttons
                button_start_y + (button_height + button_padding) * (i % column_limit),  # Y-coordinate: cycles every 'column_limit' buttons
                button_width,
                button_height
            )
            for i in range(len(self.menu_texts))]
        # Existing Button Texts
        self.in_genetic_programming = False
        self.in_SINDY = False
        self.in_back = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_genetic_programming = True
                    elif i == 1:
                        self.in_SINDY = True
                    elif i == 2:
                        self.in_check_consistency = True
                    elif i == 3:
                        self.in_back = True

        if self.in_genetic_programming:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Data_PIML_Genetic_Programming.py")])
            self.in_genetic_programming = False
        elif self.in_SINDY:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Data_PIML_SINDy.py")])
            self.in_SINDY = False
        elif self.in_back:
            self.manager.go_to(StatisticsPage())
    def exit(self):
        # Terminate the subprocess if it exists before navigating away from this page
        if self.subprocess_obj is not None and self.subprocess_obj.poll() is None:
            self.subprocess_obj.terminate()

        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Physics Informed Machine Learning Statistics", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Thermodynamics Page
class ThermodynamicsPage(Page):
    def __init__(self, page_manager=None):

        self.menu_texts = ["Air Cooler Design","Ammonia Gas Storage",
                            "Combustion","Compressor Power",
                            "Ethylene Expansion","Isentropic Air Compression",
                            "Isentropic Oxygen Compression","Joule-Thomson Methane",
                            "Liquid Nitrogen","Material Properties",
                            "Multi Component Flash","Back"]
        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        column_limit = 5  # Number of buttons per column
        number_of_columns = -(-len(self.menu_texts) // column_limit)  # Ceiling division to get the number of columns needed
        
        self.menu_rects = [
            pygame.Rect(
                button_start_x + (button_width + column_padding) * (i // column_limit),  # X-coordinate: moves to the next column every 'column_limit' buttons
                button_start_y + (button_height + button_padding) * (i % column_limit),  # Y-coordinate: cycles every 'column_limit' buttons
                button_width,
                button_height
            )
            for i in range(len(self.menu_texts))]

        self.in_air_cooler_design = False
        self.in_ammonia_gas_storage = False
        self.in_combustion = False
        self.in_compressor_power = False
        self.in_ethylene_expansion = False
        self.in_isentropic_air_compression = False
        self.in_isentropic_oxygen_compression = False
        self.in_joule_thomson_methane = False
        self.in_liquid_nitrogen = False
        self.in_material_properties = False
        self.in_multi_component_flash = False
        self.in_back = False
        self.in_thermodynamics_page = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_air_cooler_design = True
                    elif i == 1:
                        self.in_ammonia_gas_storage = True
                    elif i == 2:
                        self.in_combustion = True
                    elif i == 3:
                        self.in_compressor_power = True
                    elif i == 4:
                        self.in_ethylene_expansion = True
                    elif i == 5:
                        self.in_isentropic_air_compression = True
                    elif i == 6:
                        self.in_isentropic_oxygen_compression = True
                    elif i == 7:
                        self.in_joule_thomson_methane = True
                    elif i == 8:
                        self.in_liquid_nitrogen = True
                    elif i == 9:
                        self.in_material_properties = True
                    elif i == 10:
                        self.in_multi_component_flash = True
                    elif i == 11:
                        self.in_back = True
                    
        if self.in_air_cooler_design:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Thermodynamics_Air_Cooler_Designer.py")])
            self.in_air_cooler_design = False
        elif self.in_ammonia_gas_storage:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Thermodynamics_Ammonia_Gas_Storage_Sizing.py")])
            self.in_ammonia_gas_storage = False
        elif self.in_combustion:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Thermodynamics_Combustion_Calculations.py")])
            self.in_combustion = False
        elif self.in_compressor_power:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Thermodynamics_Compressor_Power_Sizing.py")])
            self.in_compressor_power = False
        elif self.in_ethylene_expansion:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Thermodynamics_Ethylene_Expansion.py")])
            self.in_ethylene_expansion = False
        elif self.in_isentropic_air_compression:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Thermodynamics_Isentropic_Air_Compression.py")])
            self.in_isentropic_air_compression = False
        elif self.in_isentropic_oxygen_compression:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Thermodynamics_Isentropic_Oxygen_Compression.py")])
            self.in_isentropic_oxygen_compression = False
        elif self.in_joule_thomson_methane:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Thermodynamics_Joule_Thomson_Methane.py")])
            self.in_joule_thomson_methane = False
        elif self.in_liquid_nitrogen:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Thermodynamics_Liquid_Nitrogen_Production.py")])
            self.in_liquid_nitrogen = False
        elif self.in_material_properties:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Thermodynamics_Material_Properties.py")])
            self.in_material_properties = False
        elif self.in_multi_component_flash:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Thermodynamics_Multi_Component_Flash.py")])
            self.in_multi_component_flash = False
        elif self.in_back:
            self.manager.go_to(MainMenuPage())
            self.in_back = False
    def exit(self):
        # Terminate the subprocess if it exists before navigating away from this page
        if self.subprocess_obj is not None and self.subprocess_obj.poll() is None:
            self.subprocess_obj.terminate()        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Thermodynamics Page", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Flow Sheet Simulation Page Buttons
class FlowsheetSimulationPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height)
    ]
        self.menu_texts = [
        "New Flowsheet",
        "Load Flowsheet",
        "Back"
    ]
        self.in_new_flowsheet = False
        self.in_load_flowsheet = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_new_flowsheet = True
                    elif i == 1:
                        self.in_load_flowsheet = True
                    elif i == 2:
                        self.in_back = True
        if self.in_new_flowsheet:
            self.manager.go_to(NewFlowsheetPage(page_manager=self.manager))
        elif self.in_load_flowsheet:
            self.manager.go_to(LoadFlowsheetPage())
        elif self.in_back:
            self.manager.go_to(MainMenuPage())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Flowsheet Simulator v1.0.0", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Run Flow sheet Page
class RunFlowsheetSimulation(Page):
    def __init__(self, filename, screen, page_manager):
        super().__init__()
        print("Initializing RunFlowsheetSimulation...")
        self.page_manager = page_manager
        self.filename = filename
        self.flowsheet = Flowsheet("My Flowsheet")
        self.grid = None
        self.recently_saved = False
        self.paused = False
        self.save_as = False
        self.go_to_main_menu = False
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        self.clock = pygame.time.Clock()
        self.paused = False
        self.flowsheet_renderer = FlowsheetRenderer(self.flowsheet)
        self.placing_pump = False
        self.placing_tank = False
        self.return_to_main_menu = False  # Add this line
        tab_width = 200
        tab_height = self.screen.get_height()
        self.tab_rect = pygame.Rect(self.screen.get_width() - tab_width, 0, tab_width, tab_height)

        # Load the Flowsheet object from the file
        with open(self.filename, 'rb') as f:
            self.flowsheet = pickle.load(f)

        # Load blocks from the loaded flowsheet
        for block_instance in self.flowsheet.blocks:
            block_type = type(block_instance)
            block = {
                "type": block_type.value,
                "rect": pygame.Rect(block_instance.x, block_instance.y, 150, 50),
                "instance": block_instance,
            }
            self.block_list.append(block)

        # Define colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GRAY = (128, 128, 128, 128)

        # Define fonts
        self.font = pygame.font.SysFont(None, 48)

        # Initialize dragged_block
        self.dragged_block = None

        # Define pause menu buttons
        self.pause_buttons = [
            {"label": "Save", "rect": pygame.Rect(100, 100, 300, 50)},
            {"label": "Save As", "rect": pygame.Rect(100, 200, 300, 50)},
            {"label": "Iterator", "rect": pygame.Rect(100, 300, 300, 50)},
            {"label": "Quit to Main", "rect": pygame.Rect(100, 400, 300, 50)},
            {"label": "Quit to Desktop", "rect": pygame.Rect(100, 500, 300, 50)}
        ]
        # Block types
        self.block_types = list(BlockType)
        self.block_list = []
        # Menu
        self.menu_open = False
        self.menu_rect = pygame.Rect(100, 100, 200, 200)
        # Mouse
        self.left_click = False
        self.clicked_on_block = False
        # Initialize dragged_block
        self.dragged_block = None
    def handle_event(self, event):
        if event.type == pygame.QUIT:
            pygame.quit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.paused = not self.paused
            elif event.key == pygame.K_TAB:
                self.menu_open = not self.menu_open
                if self.menu_open:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    self.menu_rect.x = mouse_x
                    self.menu_rect.y = mouse_y
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if self.tab_rect.collidepoint(event.pos):
                # Handle tab menu clicks
                for index, button in enumerate(self.tab_buttons):
                    if button["rect"].collidepoint(event.pos):
                        if button["type"] == "Centrifugal Pump":
                            self.placing_pump = True
                        elif button["type"] == "Tank":
                            self.placing_tank = True
                        else:
                            self.placing_pump = False
                            self.placing_tank = False
                        
            elif self.placing_pump:
                # Place centrifugal pump on the flowsheet
                pump_rect = pygame.Rect(event.pos[0] - 20, event.pos[1] - 20, 40, 40)
                pump_instance = {"type": BlockType.CentrifugalPump.value, "rect": pump_rect}
                self.block_list.append(pump_instance)
                self.placing_pump = False
        if self.paused:
            # Handle pause menu events
            self.draw_pause_menu()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.pause_buttons[0]['rect'].collidepoint(event.pos):
                    # Save Has been clicked
                    
                    with open(self.filename, 'wb') as f:
                        pickle.dump(self.flowsheet, f)
                    print("Game Sucessfully Saved")
                    self.recently_saved = True
                elif self.pause_buttons[1]['rect'].collidepoint(event.pos):
                    # Save as has been clicked
                    print("Save As")
                    self.save_as = True
                elif self.pause_buttons[2]['rect'].collidepoint(event.pos):
                    # Iterator has been pressed
                    print("Iterator has been pressed")
                # Check if the user clicked the "Quit to Main" button
                elif self.pause_buttons[3]["label"] == "Quit to Main" and self.pause_buttons[3]["rect"].collidepoint(mouse_x, mouse_y):
                    print("Back to Main Menu")
                    self.return_to_main_menu = True  # Update this line

                elif self.pause_buttons[4]['rect'].collidepoint(event.pos):
                    # Quit to desktop has been pressed
                    print("Quit to desktop")
                    pygame.quit()

            # Handle save as input box events
            if self.save_as:
                # Create an input box for the filename
                input_rect = pygame.Rect(100, 300, 400, 50)
                input_text = ""
                # Draw the input box and text
                overlay = pygame.Surface((self.screen_width, self.screen_height))
                overlay.fill(self.GRAY)
                overlay.set_alpha(128)
                screen.blit(overlay, (0, 0))
                pygame.draw.rect(screen, self.WHITE, input_rect)
                pygame.display.update(input_rect)

                while self.save_as:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                self.save_as = False
                            elif event.key == pygame.K_RETURN:
                                # Save the file and return to the game
                                filename = input_text.strip() + ".pkl"
                                with open(filename, "wb") as f:
                                    pickle.dump(self.flowsheet, f)
                                print("Game saved as", filename)
                                self.save_as = False
                            elif event.key == pygame.K_BACKSPACE:
                                input_text = input_text[:-1]
                                input_surface = self.font.render(input_text, True, self.BLACK)
                                pygame.draw.rect(screen, self.WHITE, input_rect)
                                screen.blit(input_surface, (input_rect.x + 5, input_rect.y + 5))
                                pygame.draw.rect(screen, self.WHITE, input_rect, 2)
                                pygame.display.update(input_rect)

                            elif event.key not in (pygame.K_LSHIFT, pygame.K_RSHIFT):
                                input_text += event.unicode

                            input_surface = self.font.render(input_text, True, self.BLACK)
                            screen.blit(input_surface, (input_rect.x + 5, input_rect.y + 5))
                            pygame.draw.rect(screen, self.WHITE, input_rect, 2)
                            pygame.display.update(input_rect)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    self.menu_open = not self.menu_open
                    if self.menu_open:
                        mouse_x, mouse_y = pygame.mouse.get_pos()
                        self.menu_rect.x = mouse_x
                        self.menu_rect.y = mouse_y      
        if self.menu_open:
            self.draw_menu()
            if self.left_click:
                mouse_x, mouse_y = pygame.mouse.get_pos()

                for i, block_type in enumerate(self.block_types):
                    if self.menu_rect.collidepoint(mouse_x, mouse_y) and (
                        self.menu_rect.y + 10 + i * 40 <= mouse_y < self.menu_rect.y + 10 + (i + 1) * 40
                    ):
                        self.add_block(block_type, mouse_x, mouse_y)
                        self.menu_open = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.left_click = True
                    mouse_x, mouse_y = pygame.mouse.get_pos()

                    if not self.menu_open:
                        for block in self.block_list:
                            if block["rect"].collidepoint(mouse_x, mouse_y):
                                self.dragged_block = block
                                break
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.left_click = False
                    self.dragged_block = None
        else:
            if self.left_click and self.dragged_block is not None:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                self.dragged_block["rect"].x = mouse_x - self.dragged_block["rect"].width // 2
                self.dragged_block["rect"].y = mouse_y - self.dragged_block["rect"].height // 2
    def draw_pause_menu(self):
        # Get the screen dimensions
        screen_width, screen_height = self.screen.get_size()
        # Create a semi-transparent overlay
        overlay = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
        overlay.fill((128, 128, 128, 128))
        # Draw the pause menu buttons on the overlay
        for button in self.pause_buttons:
            pygame.draw.rect(overlay, self.WHITE, button["rect"])
            text = self.font.render(button["label"], True, self.BLACK)
            text_rect = text.get_rect(center=button["rect"].center)
            overlay.blit(text, text_rect)

        # Draw the overlay on the screen
        self.screen.blit(overlay, (0, 0))
    def draw_menu(self):
        pygame.draw.rect(self.screen, self.WHITE, self.menu_rect)
        for i, block_type in enumerate(self.block_types):
            text = self.font.render(block_type.value, True, self.BLACK)
            self.screen.blit(text, (self.menu_rect.x + 10, self.menu_rect.y + 10 + i * 40))
    def add_block(self, block_type, x, y):
        # Create block instance based on block_type
        if block_type == BlockType.Tank:
            block = Tank(name=f"{block_type.name}_{len(self.flowsheet.blocks)}", position=(x, y))
        elif block_type == BlockType.CentrifugalPump:
            block = CentrifugalPump(name=f"{block_type.name}_{len(self.flowsheet.blocks)}", position=(x, y))
        elif block_type == BlockType.HandValve:
            block = HandValve(name=f"{block_type.name}_{len(self.flowsheet.blocks)}", position=(x, y))
        elif block_type == BlockType.ControlValve:
            block = ControlValve(name=f"{block_type.name}_{len(self.flowsheet.blocks)}", position=(x, y))
        elif block_type == BlockType.ReliefValve:
            block = ReliefValve(name=f"{block_type.name}_{len(self.flowsheet.blocks)}", position=(x, y))
        elif block_type == BlockType.FlashTank:
            block = FlashTank(name=f"{block_type.name}_{len(self.flowsheet.blocks)}", position=(x, y))
        elif block_type == BlockType.DistillationColumn:
            block = DistillationColumn(name=f"{block_type.name}_{len(self.flowsheet.blocks)}", position=(x, y))
        elif block_type == BlockType.GeneralHeatExchanger:
            block = GeneralHeatExchanger(name=f"{block_type.name}_{len(self.flowsheet.blocks)}", position=(x, y))
        elif block_type == BlockType.STHeatExchanger:
            block = STHeatExchanger(name=f"{block_type.name}_{len(self.flowsheet.blocks)}", position=(x, y))
        elif block_type == BlockType.PlateHeatExchanger:
            block = PlateHeatExchanger(name=f"{block_type.name}_{len(self.flowsheet.blocks)}", position=(x, y))
        self.flowsheet.blocks.append(block)
        print(f"Added {block.name} at position {block.position}")  # Add this line for debugging

        block.size = (150, 50)

        block_info = {"type": block_type.value, "rect": pygame.Rect(x, y, 150, 50), "instance": block}
        self.block_list.append(block_info)
    def draw_blocks(self):
        for block in self.block_list:
            if block["type"] == BlockType.CentrifugalPump.value:
                self.flowsheet_renderer.draw_centrifugal_pump(self.screen, block["rect"].center)
                print("Drawing Pump")
            elif block["type"] == BlockType.Tank.value:
                self.flowsheet_renderer.draw_storage_tank(self.screen, block["rect"].center)
                print("Drawing Tank")
            
            else:
                pygame.draw.rect(self.screen, self.BLUE, block["rect"])
                text = self.font.render(block["instance"].name, True, self.BLACK)  # Update this line to use block's name instead of its type
                self.screen.blit(text, (block["rect"].x + 10, block["rect"].y + 10))
    def run(self):
        # Initialize pygame elements
        print("Running RunFlowsheetSimulation...")  # Debugging line
        self.menu_open = False
        self.menu_rect = pygame.Rect(100, 100, 200, 200)
        self.block_types = list(BlockType)
        self.block_list = []
        self.dragged_block = None
        self.flowsheet = Flowsheet("My Flowsheet")
        self.block_list = []

        # Load blocks from the flowsheet
        for block_instance in self.flowsheet.blocks:
            block_type = type(block_instance)
            block = {
                "type": block_type.value,
                "rect": pygame.Rect(block_instance.x, block_instance.y, 150, 50),
                "instance": block_instance,
            }
            self.block_list.append(block)
    def render(self, screen):
        self.screen.fill(self.WHITE)
        self.flowsheet_renderer.render(self.screen)

        if self.menu_open:
            self.draw_menu()

        else:
            if self.left_click and self.dragged_block is not None:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                self.dragged_block["rect"].x = mouse_x - self.dragged_block["rect"].width // 2
                self.dragged_block["rect"].y = mouse_y - self.dragged_block["rect"].height // 2


        if self.paused:
            self.draw_pause_menu()
        # Draw centrifugal pumps
        for block in self.flowsheet.blocks:
            if isinstance(block, CentrifugalPump):
                self.flowsheet_renderer.draw_centrifugal_pump(screen, block.position)
            elif isinstance(block, Tank):
                self.flowsheet_renderer.draw_storage_tank(screen, block.position[0], block.position[1], width=30, height=100, border_width=2)
        pygame.display.update()
# Define the New flosheet Page
class NewFlowsheetPage(Page):
    def __init__(self, page_manager=None):
        self.input_rect = pygame.Rect(100, 300, 400, 50)
        self.input_text = ""
        self.back_rect = pygame.Rect(100, 400, 100, 50)  # Back button rectangle
        self.back_text = "Back"
        self.in_simulation = False
        self.in_back = False
        self.new_flowsheet_created = False

    def handle_event(self, event):
        # Handle keyboard events
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                # Create a new Flowsheet object with the entered name and save it
                filename = self.input_text.strip() + ".pkl"
                print("Creating new flowsheet:", filename)
                flowsheet = Flowsheet(self.input_text.strip())
                with open(filename, "wb") as f:
                    pickle.dump(flowsheet, f)
                self.manager.go_to(RunFlowsheetSimulation(filename, screen, page_manager=self.manager))
            elif event.key == pygame.K_BACKSPACE:
                self.input_text = self.input_text[:-1]
                self.update_input_field()
            elif event.unicode.isalnum():
                self.input_text += event.unicode
                self.update_input_field()

        # Handle mouse events
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.back_rect.collidepoint(event.pos):
                self.in_back = True

        pygame.display.update()


    def update_input_field(self):
        # Update the input text surface
        input_surface = font.render(self.input_text, True, BLACK)
        pygame.draw.rect(screen, WHITE, self.input_rect)
        screen.blit(input_surface, (self.input_rect.x + 5, self.input_rect.y + 5))
        pygame.draw.rect(screen, BLACK, self.input_rect, 2)

    def render(self, pygame_screen):
        screen.fill(WHITE)

        title = font.render("New Flowsheet", True, BLACK)
        screen.blit(title, (WINDOW_SIZE[1], 100))

        input_title = font.render("Type the name of the new flowsheet and hit Enter:", True, BLACK)
        screen.blit(input_title, (100, 200))

        pygame.draw.rect(screen, WHITE, self.input_rect)
        self.update_input_field()  # Call the update_input_field() method here
        pygame.draw.rect(screen, BLACK, self.input_rect, 2)

        # Draw the back button
        pygame.draw.rect(screen, GRAY, self.back_rect)
        back_button = font.render(self.back_text, True, BLACK)
        back_button_rect = back_button.get_rect()
        back_button_rect.center = self.back_rect.center
        screen.blit(back_button, back_button_rect)

        pygame.display.update(self.input_rect)

        # Check if the "Back" button has been pressed
        if self.in_back:
            self.in_back = False
            self.manager.go_to(FlowsheetSimulationPage())
# Define Load Flowsheet Page
class LoadFlowsheetPage(Page):
    def __init__(self, page_manager=None):
        self.input_rect = pygame.Rect(100, 300, 400, 50)
        self.input_text = ""
        self.back_rect = pygame.Rect(100, 400, 100, 50)  # Back button rectangle
        self.back_text = "Back"
        self.in_simulation = False
        self.in_back = False
        self.file_loaded = False

    def handle_event(self, event):
        # Handle keyboard events
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                # Check if the entered file exists
                filename = self.input_text.strip() + ".pkl"
                if os.path.exists(filename):
                    print("Loading flowsheet:", filename)
                    self.manager.go_to(RunFlowsheetSimulation(filename, screen, page_manager=self.manager))
                else:
                    print("File does not exist")
            elif event.key == pygame.K_BACKSPACE:
                self.input_text = self.input_text[:-1]
                self.update_input_field()
            elif event.unicode.isalnum():
                self.input_text += event.unicode
                self.update_input_field()

        # Handle mouse events
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.back_rect.collidepoint(event.pos):
                self.in_back = True

        pygame.display.update()

    def update_input_field(self):
        # Update the input text surface
        input_surface = font.render(self.input_text, True, BLACK)
        pygame.draw.rect(screen, WHITE, self.input_rect)
        screen.blit(input_surface, (self.input_rect.x + 5, self.input_rect.y + 5))
        pygame.draw.rect(screen, BLACK, self.input_rect, 2)

    def render(self, pygame_screen):
        screen.fill(WHITE)

        title = font.render("Load Flowsheet", True, BLACK)
        screen.blit(title, (WINDOW_SIZE[1], 100))

        input_title = font.render('Type the name of the flowsheet (.pkl file) and hit Enter:', True, BLACK)
        screen.blit(input_title, (100, 200))
        input_title2 = font.render('(A flowsheet must be created first in "New Flowsheet" page)', True, BLACK)
        screen.blit(input_title2, (100, 250))

        pygame.draw.rect(screen, WHITE, self.input_rect)
        self.update_input_field()  # Call the update_input_field() method here
        pygame.draw.rect(screen, BLACK, self.input_rect, 2)

        # Draw the back button
        pygame.draw.rect(screen, GRAY, self.back_rect)
        back_button = font.render(self.back_text, True, BLACK)
        back_button_rect = back_button.get_rect()
        back_button_rect.center = self.back_rect.center
        screen.blit(back_button, back_button_rect)

        pygame.display.update(self.input_rect)

        # Check if the "Back" button has been pressed
        if self.in_back:
            self.in_back = False
            self.manager.go_to(FlowsheetSimulationPage())
# Define Equiptment Sizing Page
class EquipmentSizingPage(Page):
    def __init__(self,page_manager = None):
        start_x = button_start_x
        self.menu_rects = [
        pygame.Rect(start_x/2-button_width/2, button_start_y, button_width, button_height),
        pygame.Rect(start_x/2-button_width/2, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(start_x/2-button_width/2, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(start_x/2-button_width/2, button_start_y + (button_height + button_padding) * 3, button_width, button_height),
        pygame.Rect(start_x/2-button_width/2, button_start_y + (button_height + button_padding) * 4, button_width, button_height),
        pygame.Rect(start_x/2-button_width/2, button_start_y + (button_height + button_padding) * 5, button_width, button_height),
        pygame.Rect(start_x, button_start_y, button_width, button_height)
    ]
        self.menu_texts = [
        "Heat Exchangers",
        "Separations",
        "Pumps",
        "Reactors",
        "Controls",
        "Back",
        "Vessels"
    ]
        self.in_heat_exchangers= False
        self.in_separations = False
        self.in_pumps = False
        self.in_reactos = False
        self.in_controls = False
        self.in_back = False
        self.in_vessels = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_heat_exchangers = True
                    elif i == 1:
                        self.in_separations = True
                    elif i == 2:
                        self.in_pumps = True
                    elif i == 3:
                        self.in_reactos = True
                    elif i == 4:
                        self.in_controls = True
                    elif i == 5:
                        self.in_back = True
                    elif i == 6:
                        self.in_vessels = True
        if self.in_heat_exchangers:
            self.manager.go_to(HeatExchangerPage())
        elif self.in_separations:
            self.manager.go_to(SeparationsPage())
        elif self.in_pumps:
            self.manager.go_to(PumpsPage())
        elif self.in_reactos:
            self.manager.go_to(ReactorsPage())
        elif self.in_controls:
            self.manager.go_to(ControlsPage())
        elif self.in_back:
            self.manager.go_to(MainMenuPage())
        elif self.in_vessels:
            self.manager.go_to(VesselsPage())
    def render(self, pygame_screen):
# Draw the main menu
        screen.fill(WHITE)
        text = font.render("Equiptment Sizing", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Process Vessels Page
class VesselsPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x/2-button_width/2, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x/2-button_width/2, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x/2-button_width/2, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x/2-button_width/2, button_start_y + (button_height + button_padding) * 3, button_width, button_height)
    ]
        self.menu_texts = [
        "Pressure Vessels",
        "Open Tanks",
        "Ventilated Tanks",
        "Back"
    ]
        self.in_pressure_vessels= False
        self.in_open_tanks = False
        self.in_ventilated_tanks = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_pressure_vessels = True
                    elif i == 1:
                        self.in_open_tanks = True
                    elif i == 2:
                        self.in_ventilated_tanks = True
                    elif i == 3:
                        self.in_back = True
        if self.in_pressure_vessels:
            self.manager.go_to(PressureVesselsPage())
        elif self.in_open_tanks:
            self.manager.go_to(OpenTankSizingPage1())
        elif self.in_ventilated_tanks:
            self.manager.go_to(VentilatedTanksPage())
        elif self.in_back:
            self.manager.go_to(EquipmentSizingPage())
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Process Vessels", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Pressure Vessels Page
class PressureVesselsPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x/2-button_width/2, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x/2-button_width/2, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x/2-button_width/2, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x/2-button_width/2, button_start_y + (button_height + button_padding) * 3, button_width, button_height),
        pygame.Rect(button_start_x/2-button_width/2, button_start_y + (button_height + button_padding) * 4, button_width, button_height)
    ]
        self.menu_texts = [
        "Boiler Vessels",
        "Expansion Vessels",
        "Generic Horizontal",
        "Generic Vertical",
        "Back"
    ]
        self.in_boiler_vessels= False
        self.in_expansion_vessels = False
        self.in_generic_horizontal = False
        self.in_generic_vertical = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_boiler_vessels = True
                    elif i == 1:
                        self.in_expansion_vessels = True
                    elif i == 2:
                        self.in_generic_horizontal = True
                    elif i == 3:
                        self.in_generic_vertical = True
                    elif i == 4:
                        self.in_back = True
        if self.in_boiler_vessels:
            self.manager.go_to(BoilerVesselPage())
        elif self.in_expansion_vessels:
            self.manager.go_to(ExpansionVesselPage())
        elif self.in_generic_horizontal:
            self.manager.go_to(GenericHorizontalVesselPage())
        elif self.in_generic_vertical:
            self.manager.go_to(GenericVerticalVesselPage())
        elif self.in_back:
            self.manager.go_to(VesselsPage())
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Types of Process Vessels", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Pressure Vessels Page for ASME Standards
class GenericVerticalPressureVesselPageASME(Page):
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Generic Vertical Pressure Vessels ASME", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Pressure Vessels Page for SIS Standards
class GenericVerticalPressureVesselsPageSIS(Page):
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Generic Vertical Pressure Vessel SIS", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Pressure Vessels Page for EN Standards
class GenericVerticalPressureVesselsPageEN(Page):
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Generic Vertical Pressure Vessels", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Generic Horizontal Vessels Page
class GenericHorizontalVesselPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x/2-button_width/2, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x/2-button_width/2, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x/2-button_width/2, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x/2-button_width/2, button_start_y + (button_height + button_padding) * 3, button_width, button_height)
    ]
        self.menu_texts = [
        "ASME Code",
        "SIS Code",
        "EN Code",
        "Back"
    ]
        self.in_ASME= False
        self.in_SIS = False
        self.in_EN = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_ASME = True
                    elif i == 1:
                        self.in_SIS = True
                    elif i == 2:
                        self.in_EN = True
                    elif i == 3:
                        self.in_back = True
        if self.in_pressure_vessels:
            self.manager.go_to(PressureVesselsPageASME())
        elif self.in_open_tanks:
            self.manager.go_to(PressureVesselsPageSIS())
        elif self.in_ventilated_tanks:
            self.manager.go_to(PressureVesselsPageEN())
        elif self.in_back:
            self.manager.go_to(PressureVesselsPage())
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Generic Horizontal Vessels", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Generic Vertical Vessels Page
class GenericVerticalVesselPage(Page):
    def __init__(self) -> None:
        super().__init__()
        self.menu_rects = [
        pygame.Rect(button_start_x/2-button_width/2, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x/2-button_width/2, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x/2-button_width/2, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x/2-button_width/2, button_start_y + (button_height + button_padding) * 3, button_width, button_height)
    ]
        self.menu_texts = [
            "ASME Code",
            "SIS Code",
            "EN Code",
            "Back"
        ]
        self.in_ASME= False
        self.in_SIS = False
        self.in_EN = False
        self.in_back = False
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_ASME = True
                    elif i == 1:
                        self.in_SIS = True
                    elif i == 2:
                        self.in_EN = True
                    elif i == 3:
                        self.in_back = True
        if self.in_ASME:
            self.manager.go_to(GenericVerticalPressureVesselPageASME())
        elif self.in_SIS:
            self.manager.go_to(GenericVerticalPressureVesselsPageSIS())
        elif self.in_EN:
            self.manager.go_to(GenericVerticalPressureVesselsPageEN())
        elif self.in_back:
            self.manager.go_to(PressureVesselsPage())
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Generic Vertical Vessels", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Boiler Pressure Vessels Page
class BoilerVesselPage(Page):
    def __init__(self, page_manager=None):
        self.menu_texts = ["Fire Tube Boiler","Recovery Boilers","Fluidized Bed Boiler","Back"]
        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        column_limit = 5  # Number of buttons per column
        number_of_columns = -(-len(self.menu_texts) // column_limit)  # Ceiling division to get the number of columns needed
        
        self.menu_rects = [
            pygame.Rect(
                button_start_x + (button_width + column_padding) * (i // column_limit),  # X-coordinate: moves to the next column every 'column_limit' buttons
                button_start_y + (button_height + button_padding) * (i % column_limit),  # Y-coordinate: cycles every 'column_limit' buttons
                button_width,
                button_height
            )
            for i in range(len(self.menu_texts))]
        # Existing Button Texts
        self.in_fire_tubes = False
        self.in_recovery_boilers = False
        self.in_fluidized_bed_boiler = False
        self.in_back = False
        self.manager = page_manager

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_fire_tubes = True
                    elif i == 1:
                        self.in_recovery_boilers = True
                    elif i == 2:
                        self.in_back = True

        if self.in_fire_tubes:
            self.manager.go_to(FireTubeBoilerPage())
            self.in_fire_tubes = False
        elif self.in_recovery_boilers:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Data_ANOVA.py")])
            self.in_anova = False
        elif self.in_back:    
            self.manager.go_to(VesselsPage())
            self.in_back = False
    def exit(self):
        # Terminate the subprocess if it exists before navigating away from this page
        if self.subprocess_obj is not None and self.subprocess_obj.poll() is None:
            self.subprocess_obj.terminate()
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Industrial Statistics", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Expansion Vessels Page
class ExpansionVesselPage(Page):
    def __init__(self, page_manager=None):
        button_padding = 30
        self.back_rect = [
            pygame.Rect(button_start_x / 2 - button_width / 2 - 200, button_start_y + (button_height + button_padding) * 4 + 100, button_width, button_height)
        ]
        self.next_rect = [
            pygame.Rect(button_start_x / 2 + button_width / 2 + 900, button_start_y + (button_height + button_padding) * 4 + 100, button_width, button_height)

        ]

        self.back_text = ["Back"]
        self.next_text = ["Next"]
        self.input_labels = ["Temp 1 (C):", "Temp 2 (C):", " System Vol (m³):", "CAS nr:"]
        self.input_values = ["", "", "", ""]
        self.input_rects = [
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y, button_width, button_height),
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y + button_height + button_padding, button_width, button_height),
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y + 2 * button_height + 2 * button_padding, button_width, button_height),
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y + 3 * button_height + 3 * button_padding, button_width, button_height)
        ]
        self.output_labels = ["Expansion Volume (m³):"]
        self.output_values = [""]
        self.output_rects = [
            pygame.Rect(3 * button_start_x / 2 + button_width / 2, button_start_y, button_width, button_height),
        ]
        self.in_back = False
        self.in_next = False
        self.manager = page_manager
        self.active_input = None
    def handle_event(self, event):
        if event.type == pygame.QUIT:
            pygame.quit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Check if the mouse clicked inside any of the input fields
            clicked_inside_input = False
            for i, rect in enumerate(self.input_rects):
                if rect.collidepoint(event.pos):
                    self.active_input = i
                    print(self.input_labels[i])
                    clicked_inside_input = True

            if not clicked_inside_input:
                self.active_input = None

            if self.back_rect[0].collidepoint(event.pos):
                self.in_back = True
                print("Back")
            elif self.next_rect[0].collidepoint(event.pos):
                self.in_next = True
                print("Next")

        elif event.type == pygame.KEYDOWN and self.active_input is not None:
            # If an input field is active and a key is pressed, update the input value accordingly
            if event.key == pygame.K_RETURN:
                # If the Enter key is pressed, submit the input value and keep the value in the input field
                print("Input value for", self.input_labels[self.active_input], "is", self.input_values[self.active_input])
                self.active_input = None
            elif event.key == pygame.K_BACKSPACE:
                # If the Backspace key is pressed, remove the last character from the input value
                self.input_values[self.active_input] = self.input_values[self.active_input][:-1]
            elif event.unicode.isdigit() or event.unicode == '.':  # Allow only numeric input and decimal points
                # If a character key is pressed, add the character to the input value
                self.input_values[self.active_input] += event.unicode
                
        if self.in_back:
            self.manager.go_to(VesselsPage())
        elif self.in_next:
            self.manager.go_to(VesselsPage())
    def update_input_field(self):
        # Update the input text surface
        input_surface = font.render(self.input_values[self.active_input], True, BLACK)
        pygame.draw.rect(screen, WHITE, self.input_rects[self.active_input])
        screen.blit(input_surface, (self.input_rects[self.active_input].x + 5, self.input_rects[self.active_input].y + 5))
        pygame.draw.rect(screen, BLACK, self.input_rects[self.active_input], 2)
    def update_output_fields(self):
        try:
            temp2 = float(self.input_values[1])
            temp1 = float(self.input_values[0])
            system_vol = float(self.input_values[2])
            cas_nr = (self.input_values[3])

            expansion_vol = (temp2 - temp1) * system_vol * 0.0001

            self.output_values = [f"{expansion_vol:.2f}"]
        except ValueError:
            self.output_values = [""]
    def draw_storage_tank(self, screen, x, y, width, height, border_width):
        # Draw the tank body
        tank_color = (255, 255, 255)  # White
        pygame.draw.rect(screen, tank_color, pygame.Rect(x, y, width, height // 2))
        # Draw the Liquid in the tank
        tank_color = (0, 0, 255)  # Blue
        pygame.draw.rect(screen, tank_color, pygame.Rect(x, y + height // 2, width, height // 2))
        # Draw the expansion liquid in the tank
        tank_color = (173, 216, 230)  # Light Blue
        pygame.draw.rect(screen, tank_color, pygame.Rect(x, y + height // 2, width, height // 4))
        # Draw the tank border
        tank_border_color = (0, 0, 0)  # Black
        pygame.draw.rect(screen, tank_border_color, pygame.Rect(x - border_width, y - border_width, width + 2 * border_width, height + 2 * border_width), border_width * 2)  
    def draw_pipe_with_double_arrow(self,screen, start_pos, end_pos, label, label_above=True):
        # Draw the line
        pygame.draw.line(screen, BLACK, start_pos, end_pos, 2)

        # Calculate the arrowhead points
        arrow_length = 5
        angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
        arrow_angle = math.radians(30)

        def draw_arrowhead(end_pos, reverse=False):
            if reverse:
                angle_adjusted = angle + math.pi
            else:
                angle_adjusted = angle

            point_a = (end_pos[0] - arrow_length * math.cos(angle_adjusted + arrow_angle),
                    end_pos[1] - arrow_length * math.sin(angle_adjusted + arrow_angle))
            point_b = (end_pos[0] - arrow_length * math.cos(angle_adjusted - arrow_angle),
                    end_pos[1] - arrow_length * math.sin(angle_adjusted - arrow_angle))
            pygame.draw.polygon(screen, BLACK, [end_pos, point_a, point_b], 0)

        # Draw arrowheads at both ends
        draw_arrowhead(end_pos)
        draw_arrowhead(start_pos, reverse=True)

        # Draw label
        font = pygame.font.Font(None, 24)
        text = font.render(label, True, BLACK)
        text_center_x = (start_pos[0] + end_pos[0]) // 2
        text_center_y = (start_pos[1] + end_pos[1]) // 2

        # Calculate label offset
        label_offset = 15
        offset_x = label_offset * math.sin(angle)
        offset_y = -label_offset * math.cos(angle)

        if not label_above:
            offset_x = -offset_x
            offset_y = -offset_y

        text_rect = text.get_rect(center=(text_center_x + offset_x, text_center_y + offset_y))

        # Rotate text
        rotated_text = pygame.transform.rotate(text, math.degrees(-angle))
        rotated_text_rect = rotated_text.get_rect(center=text_rect.center)

        screen.blit(rotated_text, rotated_text_rect)
     # Renders The Screen
    def render(self, pygame_screen):
        # Draw the main menu
        pygame_screen.fill(WHITE)
        text = font.render("Fluid Expansion Volume", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        pygame_screen.blit(text, text_rect)

        # Draw the storage tank schematic
        tank_width = 300
        tank_height = 500
        border_width = 5
        tank_x = screen_width // 2 - tank_width//2
        tank_y = screen_height // 2 - tank_height//2

        self.draw_storage_tank(pygame_screen, tank_x, tank_y, tank_width, tank_height, border_width)
        self.draw_pipe_with_double_arrow(pygame_screen,(tank_x-tank_width/2+100,tank_y+tank_height//2),(tank_x-tank_width/2+100,tank_y+tank_height),"V2")
        self.draw_pipe_with_double_arrow(pygame_screen,(tank_x-tank_width/2+50,tank_y+tank_height//2 + tank_height//4),(tank_x-tank_width/2+50,tank_y+tank_height),"V1")
        self.draw_pipe_with_double_arrow(pygame_screen,(tank_x,tank_y+100),(tank_x+tank_width,tank_y+100),"D")        
        self.update_output_fields()
        # Draw the input field rectangle
        for i, rect in enumerate(self.input_rects):
            pygame.draw.rect(pygame_screen, WHITE, rect)
            pygame.draw.rect(pygame_screen, GRAY, rect, 2)

            # Split the label into label and unit
            label, unit = self.input_labels[i].rsplit(' ', 1)

            # Draw the input field label on the right side of the input box
            label_surface = font.render(label, True, BLACK)
            label_rect = label_surface.get_rect()
            label_rect.right = rect.left - 10  # Move the label 10 pixels to the left of the input box
            label_rect.centery = rect.centery
            pygame_screen.blit(label_surface, label_rect)

            # Draw the input field text
            text_surface = font.render(self.input_values[i], True, BLACK)
            text_rect = text_surface.get_rect()
            text_rect.centerx = rect.centerx
            text_rect.centery = rect.centery
            pygame_screen.blit(text_surface, text_rect)

            # Draw the input field unit on the left side of the input box
            unit_surface = font.render(unit, True, BLACK)
            unit_rect = unit_surface.get_rect()
            unit_rect.left = rect.right + 10  # Move the unit 10 pixels to the right of the input box
            unit_rect.centery = rect.centery
            pygame_screen.blit(unit_surface, unit_rect)
        # Draw the output field rectangles
        for i, rect in enumerate(self.output_rects):
            pygame.draw.rect(pygame_screen, WHITE, rect)
            pygame.draw.rect(pygame_screen, GRAY, rect, 2)

            # Split the label into label and unit
            label, unit = self.output_labels[i].rsplit(' ', 1)

            # Draw the output field label
            label_surface = font.render(label, True, BLACK)
            label_rect = label_surface.get_rect()
            label_rect.right = rect.left - 5  # Adjust the position
            label_rect.centery = rect.centery
            pygame_screen.blit(label_surface, label_rect)

            # Draw the output field value
            value_surface = font.render(self.output_values[i], True, BLACK)
            value_rect = value_surface.get_rect()
            value_rect.centerx = rect.centerx
            value_rect.centery = rect.centery
            pygame_screen.blit(value_surface, value_rect)

            # Draw the output field unit
            unit_surface = font.render(unit, True, BLACK)
            unit_rect = unit_surface.get_rect()
            unit_rect.left = rect.right + 5  # Adjust the position
            unit_rect.centery = rect.centery
            pygame_screen.blit(unit_surface, unit_rect)
        # Draw the back button
        pygame.draw.rect(pygame_screen, GRAY, self.back_rect[0])
        back_button = font.render(self.back_text[0], True, BLACK)
        back_button_rect = back_button.get_rect()
        back_button_rect.center = self.back_rect[0].center
        pygame_screen.blit(back_button, back_button_rect)
        # Draw the next button
        pygame.draw.rect(pygame_screen, GRAY, self.next_rect[0])
        next_button = font.render(self.next_text[0], True, BLACK)
        next_button_rect = next_button.get_rect()
        next_button_rect.center = self.next_rect[0].center
        pygame_screen.blit(next_button, next_button_rect)

        # Update the active input field, if any
        if self.active_input is not None:
            self.update_input_field()

        pygame.display.update()
        if self.in_back:
            self.manager.go_to(VesselsPage())

        # Draw the back button
        pygame.draw.rect(pygame_screen, GRAY, self.back_rect[0])
        back_button = font.render(self.back_text[0], True, BLACK)
        back_button_rect = back_button.get_rect()
        back_button_rect.center = self.back_rect[0].center
        pygame_screen.blit(back_button, back_button_rect)

        # Update the active input field, if any
        if self.active_input is not None:
            self.update_input_field()

        pygame.display.update()
# Define the Open Tanks Sizing Page 1
class OpenTankSizingPage1(Page):
    def __init__(self, page_manager=None):
        button_padding = 30
        self.back_rect = [
            pygame.Rect(button_start_x //2 - button_width //2 - 200, button_start_y + (button_height + button_padding) * 4 + 100, button_width, button_height)
        ]
        self.next_rect = [
            pygame.Rect(button_start_x //4 + screen_width*0.65, button_start_y + (button_height + button_padding) * 4 + 100, button_width, button_height)

        ]

        self.back_text = ["Back"]
        self.next_text = ["Next"]
        self.input_labels = ["In Flow (m3/s):", "Out Flow (m3/s):", "Retention Time (s):", "L/D (m/m):"]
        self.input_values = ["", "", "", ""]
        self.input_rects = [
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y, button_width, button_height),
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y + button_height + button_padding, button_width, button_height),
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y + 2 * button_height + 2 * button_padding, button_width, button_height),
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y + 3 * button_height + 3 * button_padding, button_width, button_height)
        ]
        self.output_labels = ["Req. Volume (m³):", "Length (m):", "Diameter (m):"]
        self.output_values = ["", "", ""]
        self.output_rects = [
            pygame.Rect(3 * button_start_x / 2 + button_width / 2, button_start_y, button_width, button_height),
            pygame.Rect(3 * button_start_x / 2 + button_width / 2, button_start_y + button_height + button_padding, button_width, button_height),
            pygame.Rect(3 * button_start_x / 2 + button_width / 2, button_start_y + 2 * button_height + 2 * button_padding, button_width, button_height)
        ]
        self.in_back = False
        self.in_next = False
        self.manager = page_manager
        self.active_input = None
    def handle_event(self, event):
        if event.type == pygame.QUIT:
            pygame.quit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Check if the mouse clicked inside any of the input fields
            clicked_inside_input = False
            for i, rect in enumerate(self.input_rects):
                if rect.collidepoint(event.pos):
                    self.active_input = i
                    print(self.input_labels[i])
                    clicked_inside_input = True

            if not clicked_inside_input:
                self.active_input = None

            if self.back_rect[0].collidepoint(event.pos):
                self.in_back = True
                print("Back")
            elif self.next_rect[0].collidepoint(event.pos):
                self.in_next = True
                print("Next")

        elif event.type == pygame.KEYDOWN and self.active_input is not None:
            # If an input field is active and a key is pressed, update the input value accordingly
            if event.key == pygame.K_RETURN:
                # If the Enter key is pressed, submit the input value and keep the value in the input field
                print("Input value for", self.input_labels[self.active_input], "is", self.input_values[self.active_input])
                self.active_input = None
            elif event.key == pygame.K_BACKSPACE:
                # If the Backspace key is pressed, remove the last character from the input value
                self.input_values[self.active_input] = self.input_values[self.active_input][:-1]
            elif event.unicode.isdigit() or event.unicode == '.':  # Allow only numeric input and decimal points
                # If a character key is pressed, add the character to the input value
                self.input_values[self.active_input] += event.unicode
                
        if self.in_back:
            self.manager.go_to(VesselsPage())
        elif self.in_next:
            self.manager.go_to(OpenTankSizingPage2())
    def update_input_field(self):
        # Update the input text surface
        input_surface = font.render(self.input_values[self.active_input], True, BLACK)
        pygame.draw.rect(screen, WHITE, self.input_rects[self.active_input])
        screen.blit(input_surface, (self.input_rects[self.active_input].x + 5, self.input_rects[self.active_input].y + 5))
        pygame.draw.rect(screen, BLACK, self.input_rects[self.active_input], 2)
    def update_output_fields(self):
        try:
            retention_time = float(self.input_values[2])
            out_flow = float(self.input_values[1])
            ld_ratio = float(self.input_values[3])

            req_volume = retention_time * out_flow
            diameter = (1/3)*(4 * req_volume*9 / (math.pi))**(1/3)
            length = diameter * ld_ratio

            self.output_values = [f"{req_volume:.2f}", f"{length:.2f}", f"{diameter:.2f}"]
        except ValueError:
            self.output_values = ["", "", ""]

    def draw_storage_tank(self, screen, x, y, width, height, border_width):
        # Draw the tank body
        tank_color = (255, 255, 255)  # White
        pygame.draw.rect(screen, tank_color, pygame.Rect(x, y, width, height // 2))
        # Draw the Liquid in the tank
        tank_color = (0, 0, 255)  # Blue
        pygame.draw.rect(screen, tank_color, pygame.Rect(x, y + height // 2, width, height // 2))
        tank_border_color = (0, 0, 0)  # Black
        pygame.draw.rect(screen, tank_border_color, pygame.Rect(x - border_width, y - border_width, width + 2 * border_width, height + 2 * border_width), border_width * 2)

        
    def draw_pipe_with_double_arrow(self,screen, start_pos, end_pos, label, label_above=True):
        # Draw the line
        pygame.draw.line(screen, BLACK, start_pos, end_pos, 2)

        # Calculate the arrowhead points
        arrow_length = 10
        angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
        arrow_angle = math.radians(30)

        def draw_arrowhead(end_pos, reverse=False):
            if reverse:
                angle_adjusted = angle + math.pi
            else:
                angle_adjusted = angle

            point_a = (end_pos[0] - arrow_length * math.cos(angle_adjusted + arrow_angle),
                    end_pos[1] - arrow_length * math.sin(angle_adjusted + arrow_angle))
            point_b = (end_pos[0] - arrow_length * math.cos(angle_adjusted - arrow_angle),
                    end_pos[1] - arrow_length * math.sin(angle_adjusted - arrow_angle))
            pygame.draw.polygon(screen, BLACK, [end_pos, point_a, point_b], 0)

        # Draw arrowheads at both ends
        draw_arrowhead(end_pos)
        draw_arrowhead(start_pos, reverse=True)

        # Draw label
        font = pygame.font.Font(None, 24)
        text = font.render(label, True, BLACK)
        text_center_x = (start_pos[0] + end_pos[0]) // 2
        text_center_y = (start_pos[1] + end_pos[1]) // 2

        # Calculate label offset
        label_offset = 15
        offset_x = label_offset * math.sin(angle)
        offset_y = -label_offset * math.cos(angle)

        if not label_above:
            offset_x = -offset_x
            offset_y = -offset_y

        text_rect = text.get_rect(center=(text_center_x + offset_x, text_center_y + offset_y))

        # Rotate text
        rotated_text = pygame.transform.rotate(text, math.degrees(-angle))
        rotated_text_rect = rotated_text.get_rect(center=text_rect.center)

        screen.blit(rotated_text, rotated_text_rect)

    # Renders The Screen
    def render(self, pygame_screen):
        # Draw the main menu
        pygame_screen.fill(WHITE)
        text = font.render("Storage Tank Sizing Based off Retention Time: Assumed Cylindrical", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        pygame_screen.blit(text, text_rect)

        # Draw the storage tank schematic
        tank_width = 300
        tank_height = 500
        border_width = 5
        tank_x = screen_width // 2 - tank_width//2
        tank_y = screen_height // 2 - tank_height//2

        self.draw_storage_tank(pygame_screen, tank_x, tank_y, tank_width, tank_height, border_width)
        self.draw_pipe_with_double_arrow(pygame_screen,(tank_x-tank_width/2+100,tank_y+tank_height),(tank_x-tank_width/2+100,tank_y),"L")
        self.draw_pipe_with_double_arrow(pygame_screen,(tank_x,tank_y+100),(tank_x+tank_width,tank_y+100),"D")        
        self.update_output_fields()
        # Draw the input field rectangle
        for i, rect in enumerate(self.input_rects):
            pygame.draw.rect(pygame_screen, WHITE, rect)
            pygame.draw.rect(pygame_screen, GRAY, rect, 2)

            # Split the label into label and unit
            label, unit = self.input_labels[i].rsplit(' ', 1)

            # Draw the input field label on the right side of the input box
            label_surface = font.render(label, True, BLACK)
            label_rect = label_surface.get_rect()
            label_rect.right = rect.left - 10  # Move the label 10 pixels to the left of the input box
            label_rect.centery = rect.centery
            pygame_screen.blit(label_surface, label_rect)

            # Draw the input field text
            text_surface = font.render(self.input_values[i], True, BLACK)
            text_rect = text_surface.get_rect()
            text_rect.centerx = rect.centerx
            text_rect.centery = rect.centery
            pygame_screen.blit(text_surface, text_rect)

            # Draw the input field unit on the left side of the input box
            unit_surface = font.render(unit, True, BLACK)
            unit_rect = unit_surface.get_rect()
            unit_rect.left = rect.right + 10  # Move the unit 10 pixels to the right of the input box
            unit_rect.centery = rect.centery
            pygame_screen.blit(unit_surface, unit_rect)
        # Draw the output field rectangles
        for i, rect in enumerate(self.output_rects):
            pygame.draw.rect(pygame_screen, WHITE, rect)
            pygame.draw.rect(pygame_screen, GRAY, rect, 2)

            # Split the label into label and unit
            label, unit = self.output_labels[i].rsplit(' ', 1)

            # Draw the output field label
            label_surface = font.render(label, True, BLACK)
            label_rect = label_surface.get_rect()
            label_rect.right = rect.left - 5  # Adjust the position
            label_rect.centery = rect.centery
            pygame_screen.blit(label_surface, label_rect)

            # Draw the output field value
            value_surface = font.render(self.output_values[i], True, BLACK)
            value_rect = value_surface.get_rect()
            value_rect.centerx = rect.centerx
            value_rect.centery = rect.centery
            pygame_screen.blit(value_surface, value_rect)

            # Draw the output field unit
            unit_surface = font.render(unit, True, BLACK)
            unit_rect = unit_surface.get_rect()
            unit_rect.left = rect.right + 5  # Adjust the position
            unit_rect.centery = rect.centery
            pygame_screen.blit(unit_surface, unit_rect)
        # Draw the back button
        pygame.draw.rect(pygame_screen, GRAY, self.back_rect[0])
        back_button = font.render(self.back_text[0], True, BLACK)
        back_button_rect = back_button.get_rect()
        back_button_rect.center = self.back_rect[0].center
        pygame_screen.blit(back_button, back_button_rect)
        # Draw the next button
        pygame.draw.rect(pygame_screen, GRAY, self.next_rect[0])
        next_button = font.render(self.next_text[0], True, BLACK)
        next_button_rect = next_button.get_rect()
        next_button_rect.center = self.next_rect[0].center
        pygame_screen.blit(next_button, next_button_rect)

        # Update the active input field, if any
        if self.active_input is not None:
            self.update_input_field()

        pygame.display.update()
        if self.in_back:
            self.manager.go_to(VesselsPage())

        # Draw the back button
        pygame.draw.rect(pygame_screen, GRAY, self.back_rect[0])
        back_button = font.render(self.back_text[0], True, BLACK)
        back_button_rect = back_button.get_rect()
        back_button_rect.center = self.back_rect[0].center
        pygame_screen.blit(back_button, back_button_rect)

        # Update the active input field, if any
        if self.active_input is not None:
            self.update_input_field()

        pygame.display.update()
# Define the open Tanks Sizing Page 2
class OpenTankSizingPage2(Page):
    def __init__(self, page_manager=None):
        button_padding = 30
        self.back_rect = [
            pygame.Rect(button_start_x / 2 - button_width / 2 - 200, button_start_y + (button_height + button_padding) * 4 + 100, button_width, button_height)
        ]

        self.back_text = [
            "Back"
        ]
        self.input_labels = ["Outside Temp (C):","Outside Sun [lux]","Wind Speed (m/s)", "Liq Temp (C)", "Media CAS:", "Surf. Vel. (m/s):","Insul Conductivity (W/mK):"]
        self.input_values = ["", "", "", "","","",""]
        self.input_rects = [
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y, button_width, button_height),
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y + button_height + button_padding, button_width, button_height),
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y + 2 * button_height + 2 * button_padding, button_width, button_height),
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y + 3 * button_height + 3 * button_padding, button_width, button_height)
        ]
        self.output_labels = ["Insul. Thickness [mm]:", "Heat Loss (kW):", "Heat Loss (Btu/hr):"]
        self.output_values = ["", "", ""]
        self.output_rects = [
            pygame.Rect(3 * button_start_x / 2 + button_width / 2, button_start_y, button_width, button_height),
            pygame.Rect(3 * button_start_x / 2 + button_width / 2, button_start_y + button_height + button_padding, button_width, button_height),
            pygame.Rect(3 * button_start_x / 2 + button_width / 2, button_start_y + 2 * button_height + 2 * button_padding, button_width, button_height)
        ]
        self.in_back = False
        self.manager = page_manager
        self.active_input = None
    def handle_event(self, event):
        if event.type == pygame.QUIT:
            pygame.quit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Check if the mouse clicked inside any of the input fields
            clicked_inside_input = False
            for i, rect in enumerate(self.input_rects):
                if rect.collidepoint(event.pos):
                    self.active_input = i
                    print(self.input_labels[i])
                    clicked_inside_input = True

            if not clicked_inside_input:
                self.active_input = None

            if self.back_rect[0].collidepoint(event.pos):
                self.in_back = True
                print("Back")

        elif event.type == pygame.KEYDOWN and self.active_input is not None:
            # If an input field is active and a key is pressed, update the input value accordingly
            if event.key == pygame.K_RETURN:
                # If the Enter key is pressed, submit the input value and keep the value in the input field
                print("Input value for", self.input_labels[self.active_input], "is", self.input_values[self.active_input])
                self.active_input = None
                self.update_output_fields()
            elif event.key == pygame.K_BACKSPACE:
                # If the Backspace key is pressed, remove the last character from the input value
                self.input_values[self.active_input] = self.input_values[self.active_input][:-1]
            elif event.unicode.isdigit() or event.unicode == '.':  # Allow only numeric input and decimal points
                # If a character key is pressed, add the character to the input value
                self.input_values[self.active_input] += event.unicode
                
        if self.in_back:
            self.manager.go_to(OpenTankSizingPage1())
    def update_input_field(self):
        # Update the input text surface
        input_surface = font.render(self.input_values[self.active_input], True, BLACK)
        pygame.draw.rect(screen, WHITE, self.input_rects[self.active_input])
        screen.blit(input_surface, (self.input_rects[self.active_input].x + 5, self.input_rects[self.active_input].y + 5))
        pygame.draw.rect(screen, BLACK, self.input_rects[self.active_input], 2)
    def update_output_fields(self):
        insulation_thickness_mm, heat_loss_kw, heat_loss_btuh = self.calculate_insulation_thickness_and_heat_loss()
        
        if isinstance(insulation_thickness_mm, (int, float)):
            self.output_values[0] = f"{insulation_thickness_mm:.2f}"
        else:
            self.output_values[0] = ""
            
        if isinstance(heat_loss_kw, (int, float)):
            self.output_values[1] = f"{heat_loss_kw:.2f}"
        else:
            self.output_values[1] = ""
            
        if isinstance(heat_loss_btuh, (int, float)):
            self.output_values[2] = f"{heat_loss_btuh:.2f}"
        else:
            self.output_values[2] = ""

    def calculate_insulation_thickness_and_heat_loss(self):
        try:
            T_outside = float(self.input_values[0])
            lux = float(self.input_values[1])
            wind_speed = float(self.input_values[2])
            T_liquid = float(self.input_values[3])
            media_CAS = float(self.input_values[4])
            surf_vel = float(self.input_values[5])
            insul_conductivity = float(self.input_values[6])

            # Add your specific formulas for calculating insulation thickness and heat loss
            insulation_thickness = (T_liquid - T_outside) / (wind_speed * insul_conductivity)
            heat_loss_kw = insul_conductivity * (T_liquid - T_outside) * surf_vel / insulation_thickness
            heat_loss_btuh = heat_loss_kw * 3412.142

            return insulation_thickness * 1000, heat_loss_kw, heat_loss_btuh
        except ValueError:
            return "", "", ""

    def draw_storage_tank(self,screen, x, y, width, height, border_width):
        """
        Draws a schematic of a liquid open storage tank with a hemispherical top and a blue lower half
        and a white upper half with black border around it.
        
        Arguments:
        screen -- the Pygame surface on which to draw the schematic
        x -- the x-coordinate of the top-left corner of the tank
        y -- the y-coordinate of the top-left corner of the tank
        width -- the width of the tank
        height -- the height of the tank
        border_width -- the width of the border around the tank
        """
        
        # Draw the tank body
        tank_color = (255, 255, 255) # White
        pygame.draw.rect(screen, tank_color, pygame.Rect(x, y, width, height // 2))
        tank_color = (0, 0, 255) # Blue
        pygame.draw.rect(screen, tank_color, pygame.Rect(x, y + height // 2, width, height // 2))
        tank_border_color = (0, 0, 0) # Black
        pygame.draw.rect(screen, tank_border_color, pygame.Rect(x - border_width, y - border_width, width + 2 * border_width, height + 2 * border_width), border_width * 2)
        #pygame.draw.circle(screen, tank_border_color, (x + width // 2, y), width // 2 + border_width, border_width)

    def render(self, pygame_screen):
        # Draw the main menu
        pygame_screen.fill(WHITE)
        text = font.render("Storage Tank: Assumed Cylindrical", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        pygame_screen.blit(text, text_rect)

        # Draw the storage tank schematic
        tank_width = 300
        tank_height = 500
        border_width = 5
        tank_x = screen_width // 2 - tank_width//2
        tank_y = screen_height // 2 - tank_height//2

        self.draw_storage_tank(pygame_screen, tank_x, tank_y, tank_width, tank_height, border_width)
        self.update_output_fields()
        # Draw the input field rectangle
        for i, rect in enumerate(self.input_rects):
            pygame.draw.rect(pygame_screen, WHITE, rect)
            pygame.draw.rect(pygame_screen, GRAY, rect, 2)

            # Split the label into label and unit
            label, unit = self.input_labels[i].rsplit(' ', 1)

            # Draw the input field label on the right side of the input box
            label_surface = font.render(label, True, BLACK)
            label_rect = label_surface.get_rect()
            label_rect.right = rect.left - 10  # Move the label 10 pixels to the left of the input box
            label_rect.centery = rect.centery
            pygame_screen.blit(label_surface, label_rect)

            # Draw the input field text
            text_surface = font.render(self.input_values[i], True, BLACK)
            text_rect = text_surface.get_rect()
            text_rect.centerx = rect.centerx
            text_rect.centery = rect.centery
            pygame_screen.blit(text_surface, text_rect)

            # Draw the input field unit on the left side of the input box
            unit_surface = font.render(unit, True, BLACK)
            unit_rect = unit_surface.get_rect()
            unit_rect.left = rect.right + 10  # Move the unit 10 pixels to the right of the input box
            unit_rect.centery = rect.centery
            pygame_screen.blit(unit_surface, unit_rect)

        # Draw the output field rectangles
        for i, rect in enumerate(self.output_rects):
            pygame.draw.rect(pygame_screen, WHITE, rect)
            pygame.draw.rect(pygame_screen, GRAY, rect, 2)

            # Split the label into label and unit
            label, unit = self.output_labels[i].rsplit(' ', 1)

            # Draw the output field label
            label_surface = font.render(label, True, BLACK)
            label_rect = label_surface.get_rect()
            label_rect.right = rect.left - 5  # Adjust the position
            label_rect.centery = rect.centery
            pygame_screen.blit(label_surface, label_rect)

            # Draw the output field value
            value_surface = font.render(self.output_values[i], True, BLACK)
            value_rect = value_surface.get_rect()
            value_rect.centerx = rect.centerx
            value_rect.centery = rect.centery
            pygame_screen.blit(value_surface, value_rect)

            # Draw the output field unit
            unit_surface = font.render(unit, True, BLACK)
            unit_rect = unit_surface.get_rect()
            unit_rect.left = rect.right + 5  # Adjust the position
            unit_rect.centery = rect.centery
            pygame_screen.blit(unit_surface, unit_rect)

        # Draw the back button
        pygame.draw.rect(pygame_screen, GRAY, self.back_rect[0])
        back_button = font.render(self.back_text[0], True, BLACK)
        back_button_rect = back_button.get_rect()
        back_button_rect.center = self.back_rect[0].center
        pygame_screen.blit(back_button, back_button_rect)

        # Update the active input field, if any
        if self.active_input is not None:
            self.update_input_field()

        pygame.display.update()
        if self.in_back:
            self.manager.go_to(VesselsPage())

        # Draw the back button
        pygame.draw.rect(pygame_screen, GRAY, self.back_rect[0])
        back_button = font.render(self.back_text[0], True, BLACK)
        back_button_rect = back_button.get_rect()
        back_button_rect.center = self.back_rect[0].center
        pygame_screen.blit(back_button, back_button_rect)

        # Update the active input field, if any
        if self.active_input is not None:
            self.update_input_field()

        pygame.display.update()
# Define the Ventilated Tanks Page
class VentilatedTanksPage(Page):
    def __init__(self, page_manager=None):
        button_padding = 30
        self.back_rect = [
            pygame.Rect(button_start_x / 2 - button_width / 2 - 200, button_start_y + (button_height + button_padding) * 4 + 100, button_width, button_height)
        ]
        self.next_rect = [
            pygame.Rect(button_start_x / 2 + button_width / 2 + 900, button_start_y + (button_height + button_padding) * 4 + 100, button_width, button_height)

        ]

        self.back_text = ["Back"]
        self.next_text = ["Next"]
        self.input_labels = ["In Flow (m3/s):", "Out Flow (m3/s):", "Retention Time (s):", "L/D (m/m):"]
        self.input_values = ["", "", "", ""]
        self.input_rects = [
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y, button_width, button_height),
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y + button_height + button_padding, button_width, button_height),
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y + 2 * button_height + 2 * button_padding, button_width, button_height),
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y + 3 * button_height + 3 * button_padding, button_width, button_height)
        ]
        self.output_labels = ["Req. Volume (m³):", "Length (m):", "Diameter (m):"]
        self.output_values = ["", "", ""]
        self.output_rects = [
            pygame.Rect(3 * button_start_x / 2 + button_width / 2, button_start_y, button_width, button_height),
            pygame.Rect(3 * button_start_x / 2 + button_width / 2, button_start_y + button_height + button_padding, button_width, button_height),
            pygame.Rect(3 * button_start_x / 2 + button_width / 2, button_start_y + 2 * button_height + 2 * button_padding, button_width, button_height)
        ]
        self.in_back = False
        self.in_next = False
        self.manager = page_manager
        self.active_input = None
    def handle_event(self, event):
        if event.type == pygame.QUIT:
            pygame.quit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Check if the mouse clicked inside any of the input fields
            clicked_inside_input = False
            for i, rect in enumerate(self.input_rects):
                if rect.collidepoint(event.pos):
                    self.active_input = i
                    print(self.input_labels[i])
                    clicked_inside_input = True

            if not clicked_inside_input:
                self.active_input = None

            if self.back_rect[0].collidepoint(event.pos):
                self.in_back = True
                print("Back")
            elif self.next_rect[0].collidepoint(event.pos):
                self.in_next = True
                print("Next")

        elif event.type == pygame.KEYDOWN and self.active_input is not None:
            # If an input field is active and a key is pressed, update the input value accordingly
            if event.key == pygame.K_RETURN:
                # If the Enter key is pressed, submit the input value and keep the value in the input field
                print("Input value for", self.input_labels[self.active_input], "is", self.input_values[self.active_input])
                self.active_input = None
            elif event.key == pygame.K_BACKSPACE:
                # If the Backspace key is pressed, remove the last character from the input value
                self.input_values[self.active_input] = self.input_values[self.active_input][:-1]
            elif event.unicode.isdigit() or event.unicode == '.':  # Allow only numeric input and decimal points
                # If a character key is pressed, add the character to the input value
                self.input_values[self.active_input] += event.unicode
                
        if self.in_back:
            self.manager.go_to(VesselsPage())
        elif self.in_next:
            self.manager.go_to(OpenTankSizingPage2())
    def update_input_field(self):
        # Update the input text surface
        input_surface = font.render(self.input_values[self.active_input], True, BLACK)
        pygame.draw.rect(screen, WHITE, self.input_rects[self.active_input])
        screen.blit(input_surface, (self.input_rects[self.active_input].x + 5, self.input_rects[self.active_input].y + 5))
        pygame.draw.rect(screen, BLACK, self.input_rects[self.active_input], 2)
    def update_output_fields(self):
        try:
            retention_time = float(self.input_values[2])
            out_flow = float(self.input_values[1])
            ld_ratio = float(self.input_values[3])

            req_volume = retention_time * out_flow
            diameter = (1/3)*(4 * req_volume*9 / (math.pi))**(1/3)
            length = diameter * ld_ratio

            self.output_values = [f"{req_volume:.2f}", f"{length:.2f}", f"{diameter:.2f}"]
        except ValueError:
            self.output_values = ["", "", ""]

    def draw_storage_tank(self, screen, x, y, width, height, border_width):
        # Draw the tank body
        tank_color = (255, 255, 255)  # White
        pygame.draw.rect(screen, tank_color, pygame.Rect(x, y, width, height // 2))
        # Draw the Liquid in the tank
        tank_color = (0, 0, 255)  # Blue
        pygame.draw.rect(screen, tank_color, pygame.Rect(x, y + height // 2, width, height // 2))
        tank_border_color = (0, 0, 0)  # Black
        pygame.draw.rect(screen, tank_border_color, pygame.Rect(x - border_width, y - border_width, width + 2 * border_width, height + 2 * border_width), border_width * 2)

        
    def draw_pipe_with_double_arrow(self,screen, start_pos, end_pos, label, label_above=True):
        # Draw the line
        pygame.draw.line(screen, BLACK, start_pos, end_pos, 2)

        # Calculate the arrowhead points
        arrow_length = 10
        angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
        arrow_angle = math.radians(30)

        def draw_arrowhead(end_pos, reverse=False):
            if reverse:
                angle_adjusted = angle + math.pi
            else:
                angle_adjusted = angle

            point_a = (end_pos[0] - arrow_length * math.cos(angle_adjusted + arrow_angle),
                    end_pos[1] - arrow_length * math.sin(angle_adjusted + arrow_angle))
            point_b = (end_pos[0] - arrow_length * math.cos(angle_adjusted - arrow_angle),
                    end_pos[1] - arrow_length * math.sin(angle_adjusted - arrow_angle))
            pygame.draw.polygon(screen, BLACK, [end_pos, point_a, point_b], 0)

        # Draw arrowheads at both ends
        draw_arrowhead(end_pos)
        draw_arrowhead(start_pos, reverse=True)

        # Draw label
        font = pygame.font.Font(None, 24)
        text = font.render(label, True, BLACK)
        text_center_x = (start_pos[0] + end_pos[0]) // 2
        text_center_y = (start_pos[1] + end_pos[1]) // 2

        # Calculate label offset
        label_offset = 15
        offset_x = label_offset * math.sin(angle)
        offset_y = -label_offset * math.cos(angle)

        if not label_above:
            offset_x = -offset_x
            offset_y = -offset_y

        text_rect = text.get_rect(center=(text_center_x + offset_x, text_center_y + offset_y))

        # Rotate text
        rotated_text = pygame.transform.rotate(text, math.degrees(-angle))
        rotated_text_rect = rotated_text.get_rect(center=text_rect.center)

        screen.blit(rotated_text, rotated_text_rect)

    # Renders The Screen
    def render(self, pygame_screen):
        # Draw the main menu
        pygame_screen.fill(WHITE)
        text = font.render("Storage Tank Sizing Based off Retention Time: Assumed Cylindrical", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        pygame_screen.blit(text, text_rect)

        # Draw the storage tank schematic
        tank_width = 300
        tank_height = 500
        border_width = 5
        tank_x = screen_width // 2 - tank_width//2
        tank_y = screen_height // 2 - tank_height//2

        self.draw_storage_tank(pygame_screen, tank_x, tank_y, tank_width, tank_height, border_width)
        self.draw_pipe_with_double_arrow(pygame_screen,(tank_x-tank_width/2+100,tank_y+tank_height),(tank_x-tank_width/2+100,tank_y),"L")
        self.draw_pipe_with_double_arrow(pygame_screen,(tank_x,tank_y+100),(tank_x+tank_width,tank_y+100),"D")        
        self.update_output_fields()
        # Draw the input field rectangle
        for i, rect in enumerate(self.input_rects):
            pygame.draw.rect(pygame_screen, WHITE, rect)
            pygame.draw.rect(pygame_screen, GRAY, rect, 2)

            # Split the label into label and unit
            label, unit = self.input_labels[i].rsplit(' ', 1)

            # Draw the input field label on the right side of the input box
            label_surface = font.render(label, True, BLACK)
            label_rect = label_surface.get_rect()
            label_rect.right = rect.left - 10  # Move the label 10 pixels to the left of the input box
            label_rect.centery = rect.centery
            pygame_screen.blit(label_surface, label_rect)

            # Draw the input field text
            text_surface = font.render(self.input_values[i], True, BLACK)
            text_rect = text_surface.get_rect()
            text_rect.centerx = rect.centerx
            text_rect.centery = rect.centery
            pygame_screen.blit(text_surface, text_rect)

            # Draw the input field unit on the left side of the input box
            unit_surface = font.render(unit, True, BLACK)
            unit_rect = unit_surface.get_rect()
            unit_rect.left = rect.right + 10  # Move the unit 10 pixels to the right of the input box
            unit_rect.centery = rect.centery
            pygame_screen.blit(unit_surface, unit_rect)
        # Draw the output field rectangles
        for i, rect in enumerate(self.output_rects):
            pygame.draw.rect(pygame_screen, WHITE, rect)
            pygame.draw.rect(pygame_screen, GRAY, rect, 2)

            # Split the label into label and unit
            label, unit = self.output_labels[i].rsplit(' ', 1)

            # Draw the output field label
            label_surface = font.render(label, True, BLACK)
            label_rect = label_surface.get_rect()
            label_rect.right = rect.left - 5  # Adjust the position
            label_rect.centery = rect.centery
            pygame_screen.blit(label_surface, label_rect)

            # Draw the output field value
            value_surface = font.render(self.output_values[i], True, BLACK)
            value_rect = value_surface.get_rect()
            value_rect.centerx = rect.centerx
            value_rect.centery = rect.centery
            pygame_screen.blit(value_surface, value_rect)

            # Draw the output field unit
            unit_surface = font.render(unit, True, BLACK)
            unit_rect = unit_surface.get_rect()
            unit_rect.left = rect.right + 5  # Adjust the position
            unit_rect.centery = rect.centery
            pygame_screen.blit(unit_surface, unit_rect)
        # Draw the back button
        pygame.draw.rect(pygame_screen, GRAY, self.back_rect[0])
        back_button = font.render(self.back_text[0], True, BLACK)
        back_button_rect = back_button.get_rect()
        back_button_rect.center = self.back_rect[0].center
        pygame_screen.blit(back_button, back_button_rect)
        # Draw the next button
        pygame.draw.rect(pygame_screen, GRAY, self.next_rect[0])
        next_button = font.render(self.next_text[0], True, BLACK)
        next_button_rect = next_button.get_rect()
        next_button_rect.center = self.next_rect[0].center
        pygame_screen.blit(next_button, next_button_rect)

        # Update the active input field, if any
        if self.active_input is not None:
            self.update_input_field()

        pygame.display.update()
        if self.in_back:
            self.manager.go_to(VesselsPage())

        # Draw the back button
        pygame.draw.rect(pygame_screen, GRAY, self.back_rect[0])
        back_button = font.render(self.back_text[0], True, BLACK)
        back_button_rect = back_button.get_rect()
        back_button_rect.center = self.back_rect[0].center
        pygame_screen.blit(back_button, back_button_rect)

        # Update the active input field, if any
        if self.active_input is not None:
            self.update_input_field()

        pygame.display.update()
# Define Heat Exchanger Page
class HeatExchangerPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 4, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 5, button_width, button_height)
    ]
        self.menu_texts = [
        "Shell and Tube",
        "Plate",
        "Spiral",
        "Single Tube",
        "Double Pipe",
        "Back"
    ]
        self.in_shell_tube_heat_exchanger= False
        self.in_plate_heat_exchanger = False
        self.in_spiral_heat_exchanger = False
        self.in_SingleTube_heat_exchanger = False
        self.in_double_pipe_heat_exchanger = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_shell_tube_heat_exchanger = True
                    elif i == 1:
                        self.in_plate_heat_exchanger = True
                    elif i == 2:
                        self.in_spiral_heat_exchanger = True
                    elif i == 3:
                        self.in_SingleTube_heat_exchanger = True
                    elif i == 4:
                        self.in_double_pipe_heat_exchanger = True
                    elif i == 5:
                        self.in_back = True
        if self.in_shell_tube_heat_exchanger:
            self.manager.go_to(ShellTubeHexPage())
        elif self.in_plate_heat_exchanger:
            self.manager.go_to(PlateHexPage())
        elif self.in_spiral_heat_exchanger:
            self.manager.go_to(SpiralHexPage())
        elif self.in_SingleTube_heat_exchanger:
            self.manager.go_to(SingleTubeHexPage())
        elif self.in_double_pipe_heat_exchanger:
            self.manager.go_to(DoublePipeHexPage())
        elif self.in_back:
            self.manager.go_to(EquipmentSizingPage())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Types of Heat Exchanger", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Shell and Tube Hex Page
class ShellTubeHexPage(Page):
    def __init__(self, page_manager=None):
        button_padding = 30
        self.menu_rects = [
            pygame.Rect(button_start_x-100, button_start_y, button_width+200, button_height+20),
            pygame.Rect(button_start_x-125, button_start_y + button_height + button_padding, button_width+250, button_height+20)
        ]
        self.menu_texts = [
            "Start Shell and Tube Heat Exchanger Designer By Örjan Johansson",
            "Back"
        ]

        self.back_text = ["Back"]
        
    
        self.in_start_shell_and_tube_heat_exchanger_designer_by_orjan_johansson = False
        self.in_back = False
        self.manager = page_manager
        self.subprocess_obj = None  # Store the subprocess object here
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_start_shell_and_tube_heat_exchanger_designer_by_orjan_johansson = True
                    elif i == 1:
                        self.in_back = True
        if self.in_start_shell_and_tube_heat_exchanger_designer_by_orjan_johansson:
            subprocess.run(["python3", os.path.join(current_directory,"apps/ST_HEX_Orjan_GUI.py")])
            self.in_start_shell_and_tube_heat_exchanger_designer_by_orjan_johansson = False
        elif self.in_back:
            self.manager.go_to(HeatExchangerPage())
    def exit(self):
        # Terminate the subprocess if it exists before navigating away from this page
        if self.subprocess_obj is not None and self.subprocess_obj.poll() is None:
            self.subprocess_obj.terminate()
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Shell and Tube Heat Exchanger Design an Sizing", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Plate Heat Exchanger Page
class PlateHexPage(Page):
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Plate Heat Exchangers", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Spiral Heat Exchanger Page
class SpiralHexPage(Page):
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Spirial Heat Exchanger", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define SingeTube Heat Exchanger Page
class SingleTubeHexPage(Page):
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Single Tube HEat Exchanger", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Double Pipe Heat Exchanger Page
class DoublePipeHexPage(Page):
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Jacked Single pipe Aka Double Pipe Heat Exchanger", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Separations Page
class SeparationsPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height)
    ]
        self.menu_texts = [
        "Gas",
        "Liquid",
        "Solid",
        "Back"
    ]
        self.in_gas_separations= False
        self.in_liquid_separations = False
        self.in_solid_separations = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_gas_separations = True
                    elif i == 1:
                        self.in_liquid_separations = True
                    elif i == 2:
                        self.in_solid_separations = True
                    elif i == 3:
                        self.in_back = True
                        self.in_back = True
        if self.in_gas_separations:
            self.manager.go_to(GasSeparationsPage())
        elif self.in_liquid_separations:
            self.manager.go_to(LiquidSeparationsPage())
        elif self.in_solid_separations:
            self.manager.go_to(SolidSeparationsPage())
        elif self.in_back:
            self.manager.go_to(EquipmentSizingPage())
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Select Separation Type", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define The Gas Separations Page
class GasSeparationsPage(Page):
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define The Liquid Separations Page
class LiquidSeparationsPage(Page):
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define The Solids Separations Page
class SolidSeparationsPage(Page):
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Pumps Page
class PumpsPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height),
    ]
        self.menu_texts = [
        "Centrifugal",
        "Positive Displacement",
        "Ejectors",
        "Back"
    ]
        self.in_centrifugal_pump= False
        self.in_positive_displacement_pump = False
        self.in_ejectors = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_centrifugal_pump = True
                    elif i == 1:
                        self.in_positive_displacement_pump = True
                    elif i == 2:
                        self.in_ejectors = True
                    elif i == 3:
                        self.in_back = True
        if self.in_centrifugal_pump:
            self.manager.go_to(CentrifugalPumpsPage())
        elif self.in_positive_displacement_pump:
            self.manager.go_to(PositiveDisplacementPumpsPage())
        elif self.in_ejectors:
            self.manager.go_to(EjectorPumpsPage())
        elif self.in_back:
            self.manager.go_to(EquipmentSizingPage())
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Types of Pumps", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Centrifugal Pumps Page
class CentrifugalPumpsPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 4, button_width, button_height)
    ]
        self.menu_texts = [
        "Multi-Stage",
        "Single Stage",
        "Magnetic Drive",
        "Submersible",
        "Back"
    ]
        self.in_multi_stage_pump= False
        self.in_single_stage_pump = False
        self.in_magnetic_drive_pump = False
        self.in_submersible_pump = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_multi_stage_pump = True
                    elif i == 1:
                        self.in_single_stage_pump = True
                    elif i == 2:
                        self.in_magnetic_drive_pump = True
                    elif i == 3:
                        self.in_submersible_pump = True
                    elif i == 4:
                        self.in_back = True
        if self.in_multi_stage_pump:
            self.manager.go_to(MultiStagePumpsPage())
        elif self.in_single_stage_pump:
            self.manager.go_to(SingleStagePumpsPage())
        elif self.in_magnetic_drive_pump:
            self.manager.go_to(MagneticDrivePumpsPage())
        elif self.in_submersible_pump:
            self.manager.go_to(SubmersiblePumpsPage())
        elif self.in_back:
            self.manager.go_to(PumpsPage())
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Types of Centrifugal Pumps", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Multi-Stage Centrifugal Pumps Page
class MultiStagePumpsPage(Page):
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the single stage centrifugal pumps page
class SingleStagePumpsPage(Page):
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Magnetic Pumps Page
class MagneticDrivePumpsPage(Page):
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Submersible Pumps Page
class SubmersiblePumpsPage(Page):
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Positive Displacement Pumps Page
class PositiveDisplacementPumpsPage(Page):
    def __init__(self,page_manager = None):
        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        
        # Existing Button Texts
        self.menu_texts = ["Piston",
        "Rotary lobe",
        "Progressive Cavity",
        "Gear",
        "Diaphragm",
        "Peristaltic",
        "Screw Pump",
        "Back"]
        
        # Existing Buttons
        column_limit = 5  # Number of buttons per column
        number_of_columns = -(-len(self.menu_texts) // column_limit)  # Ceiling division to get the number of columns needed

        self.menu_rects = [
            pygame.Rect(
                button_start_x + (button_width + column_padding) * (i // column_limit),  # X-coordinate: moves to the next column every 'column_limit' buttons
                button_start_y + (button_height + button_padding) * (i % column_limit),  # Y-coordinate: cycles every 'column_limit' buttons
                button_width,
                button_height
            )
            for i in range(len(self.menu_texts))
        ]
        
        


        self.in_piston_pump= False
        self.in_rotary_lobe_pump = False
        self.in_progressive_cavity_pump = False
        self.in_gear_pump = False
        self.in_diaphragm_pump = False
        self.in_peristaltic_pump = False
        self.in_screw_pump = False

        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.manager.go_to(PistonPumpPage())
                    elif i == 1:
                        self.manager.go_to(RotaryLobePumpPage())
                    elif i == 2:
                        self.manager.go_to(ProgressiveCavityPumpPage())
                    elif i == 3:
                        self.manager.go_to(GearPumpPage())
                    elif i == 4:
                        self.manager.go_to(DiaphragmPumpPage())
                    elif i == 5:
                        self.manager.go_to(PeristalticPumpPage())
                    elif i == 6:
                        self.manager.go_to(ScrewPumpPage())
                    elif i == 7:
                        self.manager.go_to(PumpsPage())  # Assuming 'Back' should go to the previous page
            self.manager.go_to(PumpsPage())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Positive Displacement Pump Menu", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Piston Pump Page
class PistonPumpPage(Page):
    pass
# Define Rotary Lobe Pump Page
class RotaryLobePumpPage(Page):
    pass
# Define Progressive Cavity Pump Page
class ProgressiveCavityPumpPage(Page):
    pass
# Define Gear Pump Page
class GearPumpPage(Page):
    pass
# Define Diaphragm Pump Page
class DiaphragmPumpPage(Page):
    pass
# Define Peristaltic Pump Page
class PeristalticPumpPage(Page):
    pass
# Define Screw Pump Page
class ScrewPumpPage(Page):
    pass
# Define Ejector Pumps Page
class EjectorPumpsPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Reactors Page
class ReactorsPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height)
    ]
        self.menu_texts = [
        "CSTR",
        "PFR",
        "PBR",
        "Back"
    ]
        self.in_cstr= False
        self.in_pfr = False
        self.in_pbr = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_cstr = True
                    elif i == 1:
                        self.in_pfr = True
                    elif i == 2:
                        self.in_pbr = True
                    elif i == 3:
                        self.in_back = True
        if self.in_cstr:
            self.manager.go_to(HeatExchangerPage())
        elif self.in_pfr:
            self.manager.go_to(SeparationsPage())
        elif self.in_pbr:
            self.manager.go_to(PumpsPage())
        elif self.in_back:
            self.manager.go_to(EquipmentSizingPage())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Reactor Selection", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Controls Page
class ControlsPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 4, button_width, button_height)
    ]
        self.menu_texts = [
        "Valves",
        "Sensors & Instr.",
        "Transfer Fn",
        "Fluid Handling",
        "Back"
    ]
        self.in_valves= False
        self.in_sensors = False
        self.in_transfer_fn = False
        self.in_fluid_handling = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_valves = True
                    elif i == 1:
                        self.in_sensors = True
                    elif i == 2:
                        self.in_transfer_fn = True
                    elif i == 3:
                        self.in_fluid_handling = True
                    elif i == 4:
                        self.in_back = True
        if self.in_valves:
            self.manager.go_to(ValvesPage())
        elif self.in_sensors:
            self.manager.go_to(SensorsPage())
        elif self.in_transfer_fn:
            self.manager.go_to(TransferFnPage())
        elif self.in_fluid_handling:
            self.manager.go_to(FluidHandlingPage())
        elif self.in_back:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_back:
            self.manager.go_to(MainMenuPage())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Controls Page", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Valves Page
class ValvesPage(Page):
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Valves!!", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Transfer Functions Page
class TransferFnPage(Page):
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Transfer Functions o_0", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Fluid Handling Page
class FluidHandlingPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 4, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 5, button_width, button_height)
    
    ]
        self.menu_texts = [
        "Pipes",
        "Bends",
        "Orifices",
        "Tees",
        "Vena Contracta",
        "Back"

    ]
        self.in_pipes= False
        self.in_bends = False
        self.in_orifices = False
        self.in_tees = False
        self.in_vena_contracta = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_pipes = True
                    elif i == 1:
                        self.in_bends = True
                    elif i == 2:
                        self.in_orifices = True
                    elif i == 3:
                        self.in_tees = True
                    elif i == 4:
                        self.in_vena_contracta = True
                    elif i == 5:
                        self.in_back = True
        if self.in_pipes:
            print("Pipes Page")
            self.manager.go_to(PipesPage())
        elif self.in_bends:
            print("Bends Page")
            self.manager.go_to(BendsPage())
        elif self.in_orifices:
            print("Orifice Page")
            self.manager.go_to(OrificePage())
        elif self.in_tees:
            print("Tees Page")
            self.manager.go_to(TeesPage())
        elif self.in_vena_contracta:
            print("Vena Contracta Page")
            self.manager.go_to(VenaContractaPage())
        elif self.in_back:
            print("Back")
            self.manager.go_to(ControlsPage())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Fluid Handling", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        for i, rect in enumerate(self.menu_rects):
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            pygame_screen.blit(text, text_rect)
# Define Pipes Page
class PipesPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 4, button_width, button_height)
    ]
        self.menu_texts = [
            "Diameter Sizing",
            "Pipe Pressure Drop",
            "Pipe Wall Thickness",
            "Pipe Heat Transfer",
            "Back"
        ]
        self.in_diameter_sizing = False
        self.in_pipe_pressure_drop = False
        self.in_pipe_wall_thickness = False
        self.in_pipe_heat_transfer = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_diameter_sizing = True
                    elif i == 1:
                        self.in_pipe_pressure_drop = True
                    elif i == 2:
                        self.in_pipe_wall_thickness = True
                    elif i == 3:
                        self.in_pipe_heat_transfer = True
                    elif i == 4:
                        self.in_back = True

        if self.in_diameter_sizing:
            print("Diameter Sizing Page")
            self.manager.go_to(PipeDiameterSizingPage())
        elif self.in_pipe_pressure_drop:
            print("Pipe Pressure Drop Page")
            self.manager.go_to(PipePressureDropPage())
        elif self.in_pipe_wall_thickness:
            print("Pipe Wall Thickness Page")
            self.manager.go_to(PipeWallThicknessPage())
        elif self.in_pipe_heat_transfer:
            print("Pipe Heat Transfer Page")
            self.manager.go_to(PipeHeatTransferPage())
        elif self.in_back:
            print("Back")
            self.manager.go_to(FluidHandlingPage())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Pipes", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        for i, rect in enumerate(self.menu_rects):
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            pygame_screen.blit(text, text_rect)
# Define Pipe Pressure Drop Page
class PipePressureDropPage(Page):
    def __init__(self,page_manager=None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height)
    ]
        self.menu_texts = [
            "Back",
            "Next"
        ]
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_back = True
                    elif i == 1:
                        self.in_next = True
        if self.in_back:
            print("Back")
            self.manager.go_to(PipesPage())
        elif self.in_next:
            print("Next")
            self.manager.go_to(PipePressureDropPage())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Pipe Pressure Drop", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        for i, rect in enumerate(self.menu_rects):
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            pygame_screen.blit(text, text_rect)
# Define Pipe Wall Thickness Page
class PipeWallThicknessPage(Page):
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Pipe Wall Thickness?", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Pipe Heat Transfer Page
class PipeHeatTransferPage(Page):
    def __init__(self,page_manager =None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height)
    ]
        self.menu_texts = [
            "Back",
            "Next"
        ]
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_back = True
                    elif i == 1:
                        self.in_next = True
        if self.in_back:
            print("Back")
            self.manager.go_to(PipesPage())
        elif self.in_next:
            print("Next")
            self.manager.go_to(PipeHeatTransferPage2())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Heat transfer in Pipe", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Pipe Heat Transfer Page 2
class PipeHeatTransferPage2(Page):
    def __init__(self,pygame_screen=None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height)
    ]
        self.menu_texts = [
            "Back",
            "Next"
        ]
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_back = True
                    elif i == 1:
                        self.in_next = True
        if self.in_back:
            print("Back")
            self.manager.go_to(PipeHeatTransferPage())
        elif self.in_next:
            print("Next")
            self.manager.go_to(PipeHeatTransferPage3())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Pipe Heat Transfer", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Pipe Diameter Sizing Page
class PipeDiameterSizingPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height)
    ]
        self.menu_texts = [
            "Back",
            "Next"
        ]
        self.input_labels = ["Flow (l/s):","Econ. Velocity (m/s):"]
        self.input_values = ["", ""]
        self.input_rects = [
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y, button_width, button_height),
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y + button_height + button_padding, button_width, button_height)]
        self.output_labels = ["Econ. Diameter (mm):","Econ. Diameter (in):"]
        self.output_values = ["", ""]
        self.output_rects = [
            pygame.Rect(3 * button_start_x / 2 + button_width / 2, button_start_y, button_width, button_height),
            pygame.Rect(3 * button_start_x / 2 + button_width / 2, button_start_y + button_height + button_padding, button_width, button_height)]
        self.in_back = False
        self.in_next = False
        self.manager = page_manager
        self.active_input = None
    def handle_event(self, event):
        if event.type == pygame.QUIT:
            pygame.quit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Check if the mouse clicked inside any of the input fields
            clicked_inside_input = False
            for i, rect in enumerate(self.input_rects):
                if rect.collidepoint(event.pos):
                    self.active_input = i
                    print(self.input_labels[i])
                    clicked_inside_input = True

            if not clicked_inside_input:
                self.active_input = None

            if self.menu_rects[0].collidepoint(event.pos):
                self.in_back = True
                print("Back")
            elif self.menu_rects[1].collidepoint(event.pos):
                self.in_next = True
                print("Next")
        elif event.type == pygame.KEYDOWN and self.active_input is not None:
            # If an input field is active and a key is pressed, update the input value accordingly
            if event.key == pygame.K_RETURN:
                # If the Enter key is pressed, submit the input value and keep the value in the input field
                print("Input value for", self.input_labels[self.active_input], "is", self.input_values[self.active_input])
                self.active_input = None
            elif event.key == pygame.K_BACKSPACE:
                # If the Backspace key is pressed, remove the last character from the input value
                self.input_values[self.active_input] = self.input_values[self.active_input][:-1]
            elif event.unicode.isdigit() or event.unicode == '.':  # Allow only numeric input and decimal points
                # If a character key is pressed, add the character to the input value
                self.input_values[self.active_input] += event.unicode     
        if self.in_back:
            self.manager.go_to(PipesPage())
        elif self.in_next:
            self.manager.go_to(PipeDiameterSizingPage2())
    def update_input_field(self):
        # Update the input text surface
        input_surface = font.render(self.input_values[self.active_input], True, BLACK)
        pygame.draw.rect(screen, WHITE, self.input_rects[self.active_input])
        screen.blit(input_surface, (self.input_rects[self.active_input].x + 5, self.input_rects[self.active_input].y + 5))
        pygame.draw.rect(screen, BLACK, self.input_rects[self.active_input], 2)
    def update_output_fields(self):
        try:
            flow = float(self.input_values[0])
            flow = flow/1000
            velocity = float(self.input_values[1])
            diameter_mm = ((flow*4/(velocity*math.pi))**(1/2))*1000
            diameter_in = diameter_mm/25.4

            self.output_values = [f"{diameter_mm:.2f}",f"{diameter_in:.2f}"]
        except ValueError:
            self.output_values = ["",""]
    def draw_storage_tank(self, screen, x, y, width, height, border_width):
        # Draw the tank body
        tank_color = (255, 255, 255)  # White
        pygame.draw.rect(screen, tank_color, pygame.Rect(x, y, width, height // 2))
        # Draw the Liquid in the tank
        tank_color = (0, 0, 255)  # Blue
        pygame.draw.rect(screen, tank_color, pygame.Rect(x, y + height // 2, width, height // 2))
        tank_border_color = (0, 0, 0)  # Black
        pygame.draw.rect(screen, tank_border_color, pygame.Rect(x - border_width, y - border_width, width + 2 * border_width, height + 2 * border_width), border_width * 2)     
    def draw_pipe_with_double_arrow(self,screen, start_pos, end_pos, label, label_above=True):
        # Draw the line
        pygame.draw.line(screen, BLACK, start_pos, end_pos, 2)

        # Calculate the arrowhead points
        arrow_length = 10
        angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
        arrow_angle = math.radians(30)

        def draw_arrowhead(end_pos, reverse=False):
            if reverse:
                angle_adjusted = angle + math.pi
            else:
                angle_adjusted = angle

            point_a = (end_pos[0] - arrow_length * math.cos(angle_adjusted + arrow_angle),
                    end_pos[1] - arrow_length * math.sin(angle_adjusted + arrow_angle))
            point_b = (end_pos[0] - arrow_length * math.cos(angle_adjusted - arrow_angle),
                    end_pos[1] - arrow_length * math.sin(angle_adjusted - arrow_angle))
            pygame.draw.polygon(screen, BLACK, [end_pos, point_a, point_b], 0)

        # Draw arrowheads at both ends
        draw_arrowhead(end_pos)
        draw_arrowhead(start_pos, reverse=True)

        # Draw label
        font = pygame.font.Font(None, 24)
        text = font.render(label, True, BLACK)
        text_center_x = (start_pos[0] + end_pos[0]) // 2
        text_center_y = (start_pos[1] + end_pos[1]) // 2

        # Calculate label offset
        label_offset = 15
        offset_x = label_offset * math.sin(angle)
        offset_y = -label_offset * math.cos(angle)

        if not label_above:
            offset_x = -offset_x
            offset_y = -offset_y

        text_rect = text.get_rect(center=(text_center_x + offset_x, text_center_y + offset_y))

        # Rotate text
        rotated_text = pygame.transform.rotate(text, math.degrees(-angle))
        rotated_text_rect = rotated_text.get_rect(center=text_rect.center)

        screen.blit(rotated_text, rotated_text_rect)
    # Renders The Screen
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Diameter Sizing: Knowns: Flow, Economic Velocity", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        for i, rect in enumerate(self.menu_rects):
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            pygame_screen.blit(text, text_rect)
        

        # Draw the storage tank schematic
        tank_width = 300
        tank_height = 500
        border_width = 5
        tank_x = screen_width // 2 - tank_width//2
        tank_y = screen_height // 2 - tank_height//2

        self.draw_storage_tank(pygame_screen, tank_x, tank_y, tank_width, tank_height, border_width)
        self.draw_pipe_with_double_arrow(pygame_screen,(tank_x-tank_width/2+100,tank_y+tank_height),(tank_x-tank_width/2+100,tank_y),"L")
        self.draw_pipe_with_double_arrow(pygame_screen,(tank_x,tank_y+100),(tank_x+tank_width,tank_y+100),"D")        
        self.update_output_fields()
        # Draw the input field rectangle
        for i, rect in enumerate(self.input_rects):
            pygame.draw.rect(pygame_screen, WHITE, rect)
            pygame.draw.rect(pygame_screen, GRAY, rect, 2)

            # Split the label into label and unit
            label, unit = self.input_labels[i].rsplit(' ', 1)

            # Draw the input field label on the right side of the input box
            label_surface = font.render(label, True, BLACK)
            label_rect = label_surface.get_rect()
            label_rect.right = rect.left - 10  # Move the label 10 pixels to the left of the input box
            label_rect.centery = rect.centery
            pygame_screen.blit(label_surface, label_rect)

            # Draw the input field text
            text_surface = font.render(self.input_values[i], True, BLACK)
            text_rect = text_surface.get_rect()
            text_rect.centerx = rect.centerx
            text_rect.centery = rect.centery
            pygame_screen.blit(text_surface, text_rect)

            # Draw the input field unit on the left side of the input box
            unit_surface = font.render(unit, True, BLACK)
            unit_rect = unit_surface.get_rect()
            unit_rect.left = rect.right + 10  # Move the unit 10 pixels to the right of the input box
            unit_rect.centery = rect.centery
            pygame_screen.blit(unit_surface, unit_rect)
        # Draw the output field rectangles
        for i, rect in enumerate(self.output_rects):
            pygame.draw.rect(pygame_screen, WHITE, rect)
            pygame.draw.rect(pygame_screen, GRAY, rect, 2)

            # Split the label into label and unit
            label, unit = self.output_labels[i].rsplit(' ', 1)

            # Draw the output field label
            label_surface = font.render(label, True, BLACK)
            label_rect = label_surface.get_rect()
            label_rect.right = rect.left - 5  # Adjust the position
            label_rect.centery = rect.centery
            pygame_screen.blit(label_surface, label_rect)

            # Draw the output field value
            value_surface = font.render(self.output_values[i], True, BLACK)
            value_rect = value_surface.get_rect()
            value_rect.centerx = rect.centerx
            value_rect.centery = rect.centery
            pygame_screen.blit(value_surface, value_rect)

            # Draw the output field unit
            unit_surface = font.render(unit, True, BLACK)
            unit_rect = unit_surface.get_rect()
            unit_rect.left = rect.right + 5  # Adjust the position
            unit_rect.centery = rect.centery
            pygame_screen.blit(unit_surface, unit_rect)
        # Draw the back button
        pygame.draw.rect(pygame_screen, GRAY, self.menu_rects[0])
        back_button = font.render(self.menu_texts[0], True, BLACK)
        back_button_rect = back_button.get_rect()
        back_button_rect.center = self.menu_rects[0].center
        pygame_screen.blit(back_button, back_button_rect)
        # Draw the next button
        pygame.draw.rect(pygame_screen, GRAY, self.menu_rects[1])
        next_button = font.render(self.menu_texts[1], True, BLACK)
        next_button_rect = next_button.get_rect()
        next_button_rect.center = self.menu_rects[1].center
        pygame_screen.blit(next_button, next_button_rect)

        # Update the active input field, if any
        if self.active_input is not None:
            self.update_input_field()

        pygame.display.update()
        if self.in_back:
            self.manager.go_to(FluidHandlingPage())

        # Update the active input field, if any
        if self.active_input is not None:
            self.update_input_field()

        pygame.display.update()     
# Define Pipe Diameter Sizing Page 2
class PipeDiameterSizingPage2(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height)]
        self.menu_texts = [
            "Back"]
        self.input_labels = ["Flow (l/s):","Diameter (mm):"]
        self.input_values = ["", ""]
        self.input_rects = [
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y, button_width, button_height),
            pygame.Rect(button_start_x / 2 - button_width / 2, button_start_y + button_height + button_padding, button_width, button_height)]
        self.output_labels = ["Velocity (m/s):"]
        self.output_values = [""]
        self.output_rects = [pygame.Rect(3 * button_start_x / 2 + button_width / 2, button_start_y, button_width, button_height)]
        self.in_back = False
        self.in_next = False
        self.manager = page_manager
        self.active_input = None
    def handle_event(self, event):
        if event.type == pygame.QUIT:
            pygame.quit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Check if the mouse clicked inside any of the input fields
            clicked_inside_input = False
            for i, rect in enumerate(self.input_rects):
                if rect.collidepoint(event.pos):
                    self.active_input = i
                    print(self.input_labels[i])
                    clicked_inside_input = True

            if not clicked_inside_input:
                self.active_input = None

            if self.menu_rects[0].collidepoint(event.pos):
                self.in_back = True
                print("Back")

        elif event.type == pygame.KEYDOWN and self.active_input is not None:
            # If an input field is active and a key is pressed, update the input value accordingly
            if event.key == pygame.K_RETURN:
                # If the Enter key is pressed, submit the input value and keep the value in the input field
                print("Input value for", self.input_labels[self.active_input], "is", self.input_values[self.active_input])
                self.active_input = None
            elif event.key == pygame.K_BACKSPACE:
                # If the Backspace key is pressed, remove the last character from the input value
                self.input_values[self.active_input] = self.input_values[self.active_input][:-1]
            elif event.unicode.isdigit() or event.unicode == '.':  # Allow only numeric input and decimal points
                # If a character key is pressed, add the character to the input value
                self.input_values[self.active_input] += event.unicode
                
        if self.in_back:
            self.manager.go_to(PipeDiameterSizingPage())
        elif self.in_next:
            self.manager.go_to(PipesPage())
    def update_input_field(self):
        # Update the input text surface
        input_surface = font.render(self.input_values[self.active_input], True, BLACK)
        pygame.draw.rect(screen, WHITE, self.input_rects[self.active_input])
        screen.blit(input_surface, (self.input_rects[self.active_input].x + 5, self.input_rects[self.active_input].y + 5))
        pygame.draw.rect(screen, BLACK, self.input_rects[self.active_input], 2)
    def update_output_fields(self):
        try:
            flow = float(self.input_values[0])
            flow = flow/1000
            diameter_mm = float(self.input_values[1])
            velocity = flow*4/(math.pi*(diameter_mm/1000)**2)

            self.output_values = [f"{velocity:.2f}"]
        except ValueError:
            self.output_values = [""]
    def draw_storage_tank(self, screen, x, y, width, height, border_width):
        # Draw the tank body
        tank_color = (255, 255, 255)  # White
        pygame.draw.rect(screen, tank_color, pygame.Rect(x, y, width, height // 2))
        # Draw the Liquid in the tank
        tank_color = (0, 0, 255)  # Blue
        pygame.draw.rect(screen, tank_color, pygame.Rect(x, y + height // 2, width, height // 2))
        tank_border_color = (0, 0, 0)  # Black
        pygame.draw.rect(screen, tank_border_color, pygame.Rect(x - border_width, y - border_width, width + 2 * border_width, height + 2 * border_width), border_width * 2)     
    def draw_pipe_with_double_arrow(self,screen, start_pos, end_pos, label, label_above=True):
        # Draw the line
        pygame.draw.line(screen, BLACK, start_pos, end_pos, 2)

        # Calculate the arrowhead points
        arrow_length = 10
        angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
        arrow_angle = math.radians(30)

        def draw_arrowhead(end_pos, reverse=False):
            if reverse:
                angle_adjusted = angle + math.pi
            else:
                angle_adjusted = angle

            point_a = (end_pos[0] - arrow_length * math.cos(angle_adjusted + arrow_angle),
                    end_pos[1] - arrow_length * math.sin(angle_adjusted + arrow_angle))
            point_b = (end_pos[0] - arrow_length * math.cos(angle_adjusted - arrow_angle),
                    end_pos[1] - arrow_length * math.sin(angle_adjusted - arrow_angle))
            pygame.draw.polygon(screen, BLACK, [end_pos, point_a, point_b], 0)

        # Draw arrowheads at both ends
        draw_arrowhead(end_pos)
        draw_arrowhead(start_pos, reverse=True)

        # Draw label
        font = pygame.font.Font(None, 24)
        text = font.render(label, True, BLACK)
        text_center_x = (start_pos[0] + end_pos[0]) // 2
        text_center_y = (start_pos[1] + end_pos[1]) // 2

        # Calculate label offset
        label_offset = 15
        offset_x = label_offset * math.sin(angle)
        offset_y = -label_offset * math.cos(angle)

        if not label_above:
            offset_x = -offset_x
            offset_y = -offset_y

        text_rect = text.get_rect(center=(text_center_x + offset_x, text_center_y + offset_y))

        # Rotate text
        rotated_text = pygame.transform.rotate(text, math.degrees(-angle))
        rotated_text_rect = rotated_text.get_rect(center=text_rect.center)

        screen.blit(rotated_text, rotated_text_rect)
    # Renders The Screen
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Diameter Sizing: Knowns: Flow, Pipe Diameter", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        for i, rect in enumerate(self.menu_rects):
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            pygame_screen.blit(text, text_rect)
        

        # Draw the storage tank schematic
        tank_width = 300
        tank_height = 500
        border_width = 5
        tank_x = screen_width // 2 - tank_width//2
        tank_y = screen_height // 2 - tank_height//2

        self.draw_storage_tank(pygame_screen, tank_x, tank_y, tank_width, tank_height, border_width)
        self.draw_pipe_with_double_arrow(pygame_screen,(tank_x-tank_width/2+100,tank_y+tank_height),(tank_x-tank_width/2+100,tank_y),"L")
        self.draw_pipe_with_double_arrow(pygame_screen,(tank_x,tank_y+100),(tank_x+tank_width,tank_y+100),"D")        
        self.update_output_fields()
        # Draw the input field rectangle
        for i, rect in enumerate(self.input_rects):
            pygame.draw.rect(pygame_screen, WHITE, rect)
            pygame.draw.rect(pygame_screen, GRAY, rect, 2)

            # Split the label into label and unit
            label, unit = self.input_labels[i].rsplit(' ', 1)

            # Draw the input field label on the right side of the input box
            label_surface = font.render(label, True, BLACK)
            label_rect = label_surface.get_rect()
            label_rect.right = rect.left - 10  # Move the label 10 pixels to the left of the input box
            label_rect.centery = rect.centery
            pygame_screen.blit(label_surface, label_rect)

            # Draw the input field text
            text_surface = font.render(self.input_values[i], True, BLACK)
            text_rect = text_surface.get_rect()
            text_rect.centerx = rect.centerx
            text_rect.centery = rect.centery
            pygame_screen.blit(text_surface, text_rect)

            # Draw the input field unit on the left side of the input box
            unit_surface = font.render(unit, True, BLACK)
            unit_rect = unit_surface.get_rect()
            unit_rect.left = rect.right + 10  # Move the unit 10 pixels to the right of the input box
            unit_rect.centery = rect.centery
            pygame_screen.blit(unit_surface, unit_rect)
        # Draw the output field rectangles
        for i, rect in enumerate(self.output_rects):
            pygame.draw.rect(pygame_screen, WHITE, rect)
            pygame.draw.rect(pygame_screen, GRAY, rect, 2)

            # Split the label into label and unit
            label, unit = self.output_labels[i].rsplit(' ', 1)

            # Draw the output field label
            label_surface = font.render(label, True, BLACK)
            label_rect = label_surface.get_rect()
            label_rect.right = rect.left - 5  # Adjust the position
            label_rect.centery = rect.centery
            pygame_screen.blit(label_surface, label_rect)

            # Draw the output field value
            value_surface = font.render(self.output_values[i], True, BLACK)
            value_rect = value_surface.get_rect()
            value_rect.centerx = rect.centerx
            value_rect.centery = rect.centery
            pygame_screen.blit(value_surface, value_rect)

            # Draw the output field unit
            unit_surface = font.render(unit, True, BLACK)
            unit_rect = unit_surface.get_rect()
            unit_rect.left = rect.right + 5  # Adjust the position
            unit_rect.centery = rect.centery
            pygame_screen.blit(unit_surface, unit_rect)
        # Draw the back button
        pygame.draw.rect(pygame_screen, GRAY, self.menu_rects[0])
        back_button = font.render(self.menu_texts[0], True, BLACK)
        back_button_rect = back_button.get_rect()
        back_button_rect.center = self.menu_rects[0].center
        pygame_screen.blit(back_button, back_button_rect)
        # Update the active input field, if any
        if self.active_input is not None:
            self.update_input_field()

        pygame.display.update()
        if self.in_back:
            self.manager.go_to(PipeDiameterSizingPage())

        # Update the active input field, if any
        if self.active_input is not None:
            self.update_input_field()

        pygame.display.update()
# Define Pipe Velocity Page
class PipeVelocityPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height)
    ]
        self.menu_texts = [
            "Back",
            "Next"
        ]
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_back = True
                    elif i == 1:
                        self.in_next = True
        if self.in_back:
            print("Back")
            self.manager.go_to(PipeDiameterSizingPage())
        elif self.in_next:
            print("Calculate")
            self.manager.go_to(PipeVelocityPage())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Average Pipe Velocity", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Bends Page
class BendsPage(Page):
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Orifices Page
class OrificePage(Page):
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Tees Page
class TeesPage(Page):
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Vena Contracta Page
class VenaContractaPage(Page):
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Sensors Page
class SensorsPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 4, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 5, button_width, button_height)
    ]
        self.menu_texts = [
        "Temperature",
        "Pressure",
        "Flow",
        "Level",
        "Composition",
        "Back"
    ]
        self.in_temp_sensor= False
        self.in_pressure_sensor = False
        self.in_flow_sensor = False
        self.in_level_sensor = False
        self.in_composition_sensor = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_temp_sensor = True
                    elif i == 1:
                        self.in_pressure_sensor = True
                    elif i == 2:
                        self.in_flow_sensor = True
                    elif i == 3:
                        self.in_level_sensor = True
                    elif i == 4:
                        self.in_composition_sensor = True
                    elif i == 5:
                        self.in_back = True
        if self.in_temp_sensor:
            self.manager.go_to(TemperatureSensorsPage())
        elif self.in_pressure_sensor:
            self.manager.go_to(PressureSensorsPage())
        elif self.in_flow_sensor:
            self.manager.go_to(FlowSensorsPage())
        elif self.in_level_sensor:
            self.manager.go_to(LevelSensorsPage())
        elif self.in_composition_sensor:
            self.manager.go_to(CompositionSensorsPage())
        elif self.in_back:
            self.manager.go_to(EquipmentSizingPage())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Main Types of Sensors", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        for i, rect in enumerate(self.menu_rects):
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            pygame_screen.blit(text, text_rect)
# Define Temperature Sensors Page
class TemperatureSensorsPage(Page):
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Pressure Sensors Page
class PressureSensorsPage(Page):
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Flow Sensors Page
class FlowSensorsPage(Page):
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Level Sensors Page
class LevelSensorsPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Chemical Composition Sensors Page
class CompositionSensorsPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height)
    ]
        self.menu_texts = [
        "pH",
        "Conductivity",
        "FT-IR",
        "Back"
    ]
        self.in_ph_sensor= False
        self.in_conductivity_sensor = False
        self.in_ft_ir_sensor = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_ph_sensor = True
                    elif i == 1:
                        self.in_conductivity_sensor = True
                    elif i == 2:
                        self.in_ft_ir_sensor = True
                    elif i == 3:
                        self.in_back = True
        if self.in_ph_sensor:
            self.manager.go_to(pHSensorsPage())
        elif self.in_conductivity_sensor:
            self.manager.go_to(ConductivitySensorsPage())
        elif self.in_ft_ir_sensor:
            self.manager.go_to(FtIrPage())
        elif self.in_back:
            self.manager.go_to(SensorsPage())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Types of Composition Analyzers", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        for i, rect in enumerate(self.menu_rects):
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            pygame_screen.blit(text, text_rect)
# Define the pH Sensors Page
class pHSensorsPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Conductivity Sensors Page
class ConductivitySensorsPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define FT-IR Sensors Page
class FtIrPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Gas Alarms Page
class GasAlarmsPage(Page):
    def __init__(self) -> None:
        super().__init__()
        self.menu_rects = []
        self.menu_texts = []
        self.in_back = False
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_back = True
        if self.in_back:
            self.manager.go_to(EquipmentSizingPage())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Gas Alarms", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        for i, rect in enumerate(self.menu_rects):
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Flare System Page
class FlareSystemPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Capital Cost Page
class ProcessEconomicsPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x-100, button_start_y, button_width+200, button_height+20),
        pygame.Rect(button_start_x-125, button_start_y + button_height + button_padding, button_width+250, button_height+20),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height)
    ]
        self.menu_texts = [
        "Estimate Flowsheet Capital Cost",
        "Edit Capital Cost Estimation Factors",
        "Optimal Pipe Diameter",
        "Back"
    ]
        self.in_estimate_flowsheet_capital_cost= False
        self.in_edit_capital_cost_estimation_factors = False
        self.in_optimal_pipe_diameter = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_estimate_flowsheet_capital_cost = True
                    elif i == 1:
                        self.in_edit_capital_cost_estimation_factors = True
                    elif i == 2:
                        self.in_optimal_pipe_diameter = True
                    elif i == 3:
                        self.in_back = True
        if self.in_estimate_flowsheet_capital_cost:
            self.manager.go_to(EstimateFlowsheetCapitalCostEstimationPage())
        elif self.in_edit_capital_cost_estimation_factors:
            self.manager.go_to(EditCapitalCostEstimationParametersPage())
        elif self.in_optimal_pipe_diameter:
            self.manager.go_to(OptimalPipeDiameterPage())
        elif self.in_back:
            self.manager.go_to(MainMenuPage())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Process Economics", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define The Estimate 
class OptimalPipeDiameterPage(Page):
    def __init__(self) -> None:
        self.menu_rects = [
        pygame.Rect(button_start_x-100, button_start_y, button_width+200, button_height+20),
        pygame.Rect(button_start_x-125, button_start_y + button_height + button_padding, button_width+250, button_height+20)
    ]
        self.menu_texts = [
        "Start Optimal Pipe Diameter Calculation Program for Sweden By Oscar Rexid",
        "Back"
    ]
        self.in_start_optimal_pipe_diameter_calculation_program_for_sweden_by_oscar_rexxit = False
        self.in_back = False
        self.manager = page_manager
        self.subprocess_obj = None  # Store the subprocess object here
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_start_optimal_pipe_diameter_calculation_program_for_sweden_by_oscar_rexxit = True
                    elif i == 1:
                        self.in_back = True
        if self.in_start_optimal_pipe_diameter_calculation_program_for_sweden_by_oscar_rexxit:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Pipe_Cost_Optimization.py")])
            self.in_start_optimal_pipe_diameter_calculation_program_for_sweden_by_oscar_rexxit = False
        elif self.in_back:
            self.manager.go_to(ProcessEconomicsPage())
    def exit(self):
        # Terminate the subprocess if it exists before navigating away from this page
        if self.subprocess_obj is not None and self.subprocess_obj.poll() is None:
            self.subprocess_obj.terminate()
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Optimal Pipe Diameter", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        for i, rect in enumerate(self.menu_rects):
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            pygame_screen.blit(text, text_rect)
# Define Estimate Flowsheet Capital Cost Estimation Page
class EstimateFlowsheetCapitalCostEstimationPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Edit Capital Cost Estimation Parameters Page
class EditCapitalCostEstimationParametersPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Process Safety Page
class ProcessSafetyPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x-(button_width)//8, button_start_y, button_width*1.25, button_height),
        pygame.Rect(button_start_x-(button_width)//4, button_start_y + button_height + button_padding, button_width*1.5, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height)
    ]
        self.menu_texts = process_safety_submenu_texts = [
        "Estimate Flowsheet Safety",
        "Find Chemical Safety Properties",
        "Safety Instr.",
        "Back"
    ]
        self.in_estimate_flowsheet_safety= False
        self.in_find_chemical_safety = False
        self.in_safety_instrumentation = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_estimate_flowsheet_safety = True
                    elif i == 1:
                        self.in_find_chemical_safety = True
                    elif i == 2:
                        self.in_safety_instrumentation = True
                    elif i == 3:
                        self.in_back = True
        if self.in_estimate_flowsheet_safety:
            self.manager.go_to(EstimateFlowsheetSafetyPage())
        elif self.in_find_chemical_safety:
            self.manager.go_to(FindChemicalSafetyPage())
        elif self.in_safety_instrumentation:
            self.manager.go_to(SafetyInstrumentationPage())
        elif self.in_back:
            self.manager.go_to(MainMenuPage())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Process Safety", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Flowsheet Safety Estimation Page
class EstimateFlowsheetSafetyPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Chemical Safety Page
class FindChemicalSafetyPage(Page):
    def __init__(self, page_manager=None):
        button_padding = 30
        self.menu_rects = [
            pygame.Rect(button_start_x-100, button_start_y, button_width+200, button_height+20),
            pygame.Rect(button_start_x-125, button_start_y + button_height + button_padding, button_width+250, button_height+20)
        ]
        self.menu_texts = [
            "Start Industrial Hygeine Finder",
            "Back"
        ]

        self.back_text = ["Back"]
        
    
        self.in_start_industrial_hygeine_finder = False
        self.in_back = False
        self.manager = page_manager
        self.subprocess_obj = None  # Store the subprocess object here
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_start_industrial_hygeine_finder = True
                    elif i == 1:
                        self.in_back = True
        if self.in_start_industrial_hygeine_finder:
            subprocess.run(["python3", os.path.join(current_directory,"apps/Safety_Chemical_Safety.py")])
            self.in_start_industrial_hygeine_finder = False
        elif self.in_back:
            self.manager.go_to(ProcessSafetyPage())
    def exit(self):
        # Terminate the subprocess if it exists before navigating away from this page
        if self.subprocess_obj is not None and self.subprocess_obj.poll() is None:
            self.subprocess_obj.terminate()
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("TWA STEL Data", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Safety Instrument Page
class SafetyInstrumentationPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height)
    ]
        self.menu_texts = [
        "Rupture Disks",
        "relief Valves",
        "Gas Alarms",
        "Back"
    ]
        self.in_rupture_disk= False
        self.in_relief_valves = False
        self.in_gas_alarms = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_rupture_disk = True
                    elif i == 1:
                        self.in_relief_valves = True
                    elif i == 2:
                        self.in_gas_alarms = True
                    elif i == 3:
                        self.in_back = True
        if self.in_rupture_disk:
            self.manager.go_to(RuptureDiskPage())
        elif self.in_relief_valves:
            self.manager.go_to(ReliefValvesPage())
        elif self.in_gas_alarms:
            self.manager.go_to(GasAlarmsPage())
        elif self.in_back:
            self.manager.go_to(ProcessSafetyPage())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Equipment Sizing", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        for i, rect in enumerate(self.menu_rects):
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            pygame_screen.blit(text, text_rect)
# Define Rupture Disk Page
class RuptureDiskPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define relief Valves Page
class ReliefValvesPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define Physical Properties Page
class PhysicalPropertiesPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height)
    ]
        self.menu_texts = [
        "Add a Chemical",
        "Find a Chemical",
        "Back"
    ]
        self.in_add_chemical= False
        self.in_find_chemical = False
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_add_chemical = True
                    elif i == 1:
                        self.in_find_chemical = True
                    elif i == 2:
                        self.in_back = True
                        self.in_back = True
        if self.in_add_chemical:
            screen_dimensions = (WINDOW_SIZE[0], WINDOW_SIZE[1])
            self.manager.go_to(AddChemicalPage())
        elif self.in_find_chemical:
            self.manager.go_to(FindChemicalPage())
        elif self.in_back:
            self.manager.go_to(MainMenuPage())

    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Physical Properties of Chemicals", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define a basic button class
class Button:
    def __init__(self, x, y, width, height, text, font_size, bg_color):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = pygame.font.SysFont(None, font_size)
        self.bg_color = bg_color
        self.text_color = BLACK

    def draw(self, surface):
        pygame.draw.rect(surface, self.bg_color, self.rect)
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

    def is_clicked(self, mouse_pos):
        return self.rect.collidepoint(mouse_pos)
# Define InputFieldWithUnits Class
class InputFieldWithUnits:
    def __init__(self, font, font_name, label='', max_length=None, var_name='', units=''):
        self.label = label
        self.var_name = var_name
        self.units = units
        self.max_length = max_length
        self.font = font
        self.bold_font = pygame.font.SysFont(font_name, font.get_height(), bold=True)  # Create a bold font using font_name
        self.rect = pygame.Rect(0, 0, 0, 60)
        self.active = False
        self.update_surface()

    def render(self, surface, pos):
        self.update_surface()
        self.rect = self.surface.get_rect(center=pos)
        surface.blit(self.surface, self.rect)


    def get_text(self):
        return self.label

    def add_character(self, char):
        if self.max_length is not None and len(self.label) >= self.max_length:
            return
        self.label += char
        self.update_surface()

    def remove_character(self):
        self.label = self.label[:-1]
        self.update_surface()

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.active = True
            self.label = ""
            self.update_surface()
        elif event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                self.active = False
            elif event.key == pygame.K_BACKSPACE:
                self.remove_character()
            elif event.unicode.isprintable():
                self.add_character(event.unicode)

    def update_surface(self):
        fixed_surface_width = 300

        # Use the bold_font for rendering the variable name
        var_name_surf = self.bold_font.render(self.var_name, True, BLACK)
        self.surface = pygame.Surface((fixed_surface_width, 60), pygame.SRCALPHA)

        fixed_surface_width = 500
        box_start_x = 150
        var_name_surf = self.font.render(self.var_name, True, BLACK)

        units_numerator, units_denominator = self.units.split('/')
        units_numerator_surf = self.font.render(units_numerator, True, BLACK)
        units_denominator_surf = self.font.render(units_denominator, True, BLACK)

        self.surface = pygame.Surface((fixed_surface_width, 60), pygame.SRCALPHA)

        # Clear the surface with a transparent color
        self.surface.fill((0, 0, 0, 0))

        pygame.draw.rect(self.surface, BLACK, (box_start_x, 20, int(125 * 1.50), 40), 2)
        label_surf = self.font.render(self.label, True, BLACK)
        self.surface.blit(label_surf, (box_start_x + 5, 25))

        self.surface.blit(var_name_surf, (fixed_surface_width // 2 - var_name_surf.get_width() // 2, -5))

        units_x_offset = -80  # Change this value to move the units label closer or farther from the input box
        max_unit_width = max(units_numerator_surf.get_width(), units_denominator_surf.get_width())
        units_x = fixed_surface_width - max_unit_width - 10 + units_x_offset

        current_unit_width = max(units_numerator_surf.get_width(), units_denominator_surf.get_width())
        additional_offset = max_unit_width - current_unit_width

        self.surface.blit(units_numerator_surf, (units_x + additional_offset // 2 + (max_unit_width - units_numerator_surf.get_width()) // 2, 5))
        pygame.draw.line(self.surface, BLACK, (units_x - 5 + additional_offset // 2, 30), (units_x + max_unit_width + 5 + additional_offset // 2, 30), 1)
        self.surface.blit(units_denominator_surf, (units_x + additional_offset // 2, 35))
# Define the InputField class
class InputField:
    def __init__(self, font, label='', max_length=None):
        self.label = label
        self.max_length = max_length
        self.font = font
        self.surface = pygame.Surface((250, 40))
        self.rect = self.surface.get_rect()
        self.active = False

    def get_text(self):
        return self.label

    def add_character(self, char):
        if self.max_length is not None and len(self.label) >= self.max_length:
            return
        self.label += char
        self.update_surface()

    def remove_character(self):
        self.label = self.label[:-1]
        self.update_surface()

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.active = True
        elif event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                self.active = False
            elif event.key == pygame.K_BACKSPACE:
                self.remove_character()
            elif event.unicode.isprintable():
                self.add_character(event.unicode)

    def update_surface(self):
        self.surface.fill(WHITE)
        pygame.draw.rect(self.surface, GRAY, (0, 0, 250, 40), 2)
        label_surf = self.font.render(self.label, True, BLACK)
        self.surface.blit(label_surf, (5, 5))

    def render(self, surface, pos, height):
        self.rect.center = pos
        self.update_surface()
        surface.blit(self.surface, self.rect)

    def render_value(self, surface, rect):
        pygame.draw.rect(surface, WHITE, rect)
        pygame.draw.rect(surface, GRAY, rect, 2)

        # Split the label into label and unit
        label, unit = self.label.rsplit(' ', 1)

        # Draw the input field label on the right side of the input box
        label_surface = font.render(label, True, BLACK)
        label_rect = label_surface.get_rect()
        label_rect.right = rect.left - 10  # Move the label 10 pixels to the left of the input box
        label_rect.centery = rect.centery
        surface.blit(label_surface, label_rect)

        # Draw the input field text
        text_surface = font.render(self.label, True, BLACK)
        text_rect = text_surface.get_rect()
        text_rect.centerx = rect.centerx
        text_rect.centery = rect.centery
        surface.blit(text_surface, text_rect)

        # Draw the input field unit on the left side of the input box
        unit_surface = font.render(unit, True, BLACK)
        unit_rect = unit_surface.get_rect()
        unit_rect.left = rect.right + 10  # Move the unit 10 pixels to the right of the input box
        unit_rect.centery = rect.centery
        surface.blit(unit_surface, unit_rect)
# Define the AddChemicalPage class
class AddChemicalPage(Page):
    def __init__(self, page_manager = None):
        super().__init__()
        self.cas_entry = InputField(font,label ="CAS Number",max_length = 50)
        self.chemical_name_entry = InputField(font, label="Chemical Name", max_length = 20)
        self.molecular_formula_entry = InputField(font, label ="Molecular Formula", max_length=20)
        pos = (button_start_x, button_start_y)
        self.submit_button = Button(pos[0], pos[1], button_width+100, button_height+50, 'Submit', font_size=30, bg_color=BLUE)
        self.back_button = Button(pos[0], pos[1] + button_height + button_padding, button_width, button_height, "Back", font_size=30, bg_color=GRAY)

    def handle_event(self, event):
        # Handle events for input fields and submit button
        self.cas_entry.handle_event(event)
        self.chemical_name_entry.handle_event(event)
        self.molecular_formula_entry.handle_event(event)

        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos
            if self.submit_button.is_clicked(mouse_pos):
                # Do something when the button is clicked
                print("Entry Submitted")
            elif self.back_button.is_clicked(mouse_pos):
                self.manager.go_to(MainMenuPage())


    # Render method
    def render(self, pygame_screen):
        screen.fill(WHITE)
        text = font.render("Add Chemical Data", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)

        input_y_offset = button_start_y + button_height + button_padding
        input_fields = [(self.cas_entry, (button_start_x * 1.5, input_y_offset)),
                        (self.chemical_name_entry, (button_start_x * 1.5, input_y_offset + 75)),
                        (self.molecular_formula_entry, (button_start_x * 1.5, input_y_offset + 150))]
        for field, pos in input_fields:
            field.render(pygame_screen, pos, 100)

        pygame.draw.rect(pygame_screen, GRAY, self.submit_button.rect)
        pygame.draw.rect(pygame_screen,GRAY, self.back_button.rect)

        pygame.display.update()
# Define Find Chemical Page
class FindChemicalPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Data Pre-Processing Page
class DataPreProcessingPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Quality check and Basic Cleaning Page
class DataQualityCheckAndBasicCleaningPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the temporal consistency Page
class DataTemporalConsistencyPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Temporal Data", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Normalization and Standardization Page
class DataNormalizationAndStandardizationPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Dimensionality Reduction Page
class DataDimensionalityReductionPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Noise Reduction and Outlier Handling Page
class DataNoiseReductionAndOutlierHandlingPage(Page):
    def __init__(self, page_manager=None):

        # Incremental Button Variables
        column_padding = 200  # Horizontal distance between columns
        second_column_x = button_start_x + button_width + column_padding  # X-coordinate of the second column
        
        # Existing Buttons
        self.menu_rects = [pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(6)]
        
        # New Buttons
        self.menu_rects += [pygame.Rect(second_column_x, button_start_y + (button_height + button_padding) * i, button_width, button_height) for i in range(3)]
        
        # Existing Button Texts
        self.menu_texts = ["Flowsheet Simulation", "Equipment Sizing", "Process Economics", "Process Safety", "Physical Properties", "Quit"]
        
        # New Button Texts
        self.menu_texts += ["Data Processing", "Statistics", "Thermodynamics"]

        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
        self.in_data_processing = False
        self.in_statistics = False
        self.in_thermodynamics = False


        self.manager = page_manager
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(event.pos):
                    if i == 0:
                        self.in_flowsheet_sim = True
                    elif i == 1:
                        self.in_equiptment_sizing = True
                    elif i == 2:
                        self.in_capital_cost = True
                    elif i == 3:
                        self.in_process_safety = True
                    elif i == 4:
                        self.in_physical_properties = True
                    elif i == 5:
                        self.manager.running = False
                        self.in_quit= True
                    elif i == 6:  # Data Processing Button
                        self.manager.go_to(DataProcessingPage())
                    elif i == 7:  # Statistics Button
                        self.manager.go_to(StatisticsPage())
                    elif i == 8:  # Thermodynamics Button
                        self.manager.go_to(ThermodynamicsPage())
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(ProcessEconomicsPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            self.manager.running = False
        
    def render(self, pygame_screen):
    # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        
        for i, rect in enumerate(self.menu_rects):
            # Inflate the rect to make it slightly bigger than the text
            inflated_rect = rect.inflate(20, 10)  # Increase the width by 20 and height by 10
            
            # Draw a light grey rectangle for a 3D effect (raised button appearance)
            pygame.draw.rect(pygame_screen, (200, 200, 200), inflated_rect)  # Light grey
            
            # Draw dark grey lines at the bottom and right for a 3D effect
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomleft, inflated_rect.bottomright)  # Dark grey
            pygame.draw.line(pygame_screen, (100, 100, 100), inflated_rect.bottomright, inflated_rect.topright)  # Dark grey
            
            # Draw the main button rectangle
            pygame.draw.rect(pygame_screen, BLACK, inflated_rect, 2)
            
            # Render the text in the center of the inflated rectangle
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=inflated_rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Flowsheet Renderer Class
class FlowsheetRenderer:
    def __init__(self,flowsheet):
        self.flowsheet = flowsheet
    # Draw a line
    def draw_line(screen, start_pos, end_pos, color):
        pygame.draw.line(screen, color, start_pos, end_pos, 5)
    # Draw a rectangle
    def draw_rect(screen, rect, color):
        pygame.draw.rect(screen, color, rect)
    # Draw a filled rectangle
    def draw_filled_rect(screen, rect, color):
        pygame.draw.rect(screen, color, rect, 0)
    # Draw a circle
    def draw_circle(screen, center_pos, radius, color):
        pygame.draw.circle(screen, color, center_pos, radius, 5)
    # Draw a filled circle
    def draw_filled_circle(screen, center_pos, radius, color):
        pygame.draw.circle(screen, color, center_pos, radius, 0)
    # Draw an ellipse
    def draw_ellipse(screen, rect, color):
        pygame.draw.ellipse(screen, color, rect, 5)
    # Draw a filled ellipse
    def draw_filled_ellipse(screen, rect, color):
        pygame.draw.ellipse(screen, color, rect, 0)
    # Draw a polygon
    def draw_polygon(screen, point_list, color):
        pygame.draw.polygon(screen, color, point_list, 5)
    # Draw a filled polygon
    def draw_filled_polygon(screen, point_list, color):
        pygame.draw.polygon(screen, color, point_list, 0)
    # Draw a valve
    def draw_valve(screen, center_pos):
        pygame.draw.rect(screen, BLACK, (center_pos[0] - 20, center_pos[1] - 5, 40, 10))
    # Draw a capsule with points
    def capsule_points(a, r, n, tolerance):
        top_points = []
        bottom_points = []

        for y in np.linspace(0, r, num=n // 2):
            x_minus_term = np.sqrt(r ** 2 - y ** 2)
            x_plus_term = x_minus_term

            for x in np.linspace(-a - x_minus_term, a + x_plus_term, num=n // 2):
                equation = (x - a - x_minus_term) * (x + a + x_plus_term) * y * (y - r)

                if abs(equation) <= tolerance:
                    top_points.append((x, y))
                    bottom_points.append((x, -y))

        return top_points, bottom_points
    # Draw Capsule
    def draw_capsule(screen, center_pos, width, height, color, thickness, rotation=0):
        a = width / 2
        r = height / 2
        n = 200
        tolerance = 10 ** (-n)

        top_points, bottom_points = capsule_points(a, r, n, tolerance)

        if rotation == 90:
            translated_top_points = [(int(center_pos[0] - y), int(center_pos[1] - x)) for x, y in top_points]
            translated_bottom_points = [(int(center_pos[0] - y), int(center_pos[1] - x)) for x, y in bottom_points]
        else:
            translated_top_points = [(int(center_pos[0] + x), int(center_pos[1] - y)) for x, y in top_points]
            translated_bottom_points = [(int(center_pos[0] + x), int(center_pos[1] - y)) for x, y in bottom_points]

        pygame.draw.lines(screen, color, False, translated_top_points, thickness)
        pygame.draw.lines(screen, color, False, translated_bottom_points, thickness)
    # Draw a horizontal flash tank
    def draw_horizontal_flash_tank(screen, center_pos, label):
        # Create font and render text
        font = pygame.font.Font(None, 24)
        text = font.render(label, True, BLACK)
        text_width, text_height = text.get_size()

        # Calculate the size of the capsule shape based on the text size
        padding = 10
        capsule_width = text_width + 2 * padding
        capsule_height = text_height + 2 * padding

        # Draw capsule shape
        draw_capsule(screen, center_pos, capsule_width, capsule_height, BLACK, 5)
        draw_capsule(screen, center_pos, capsule_width-3, capsule_height-3, WHITE, 5)

        # Draw label
        text_rect = text.get_rect(center=center_pos)
        screen.blit(text, text_rect)
    # Draw a vertical flash tank
    def draw_vertical_flash_tank(screen, center_pos, label):
        # Create font and render text
        font = pygame.font.Font(None, 24)
        text = font.render(label, True, BLACK)
        text_width, text_height = text.get_size()

        # Calculate the size of the capsule shape based on the text size
        padding = 10
        capsule_width = text_height + 2 * padding
        capsule_height = text_width + 2 * padding

        # Draw capsule shape
        draw_capsule(screen, center_pos, capsule_width, capsule_height, BLACK, 5, rotation=90)
        draw_capsule(screen, center_pos, capsule_width - 3, capsule_height - 3, WHITE, 5, rotation=90)

        # Draw label
        # Rotate the text 90 degrees counterclockwise
        #rotated_text = pygame.transform.rotate(text, 90)
        text_rect = text.get_rect(center=center_pos)
        screen.blit(text, text_rect)
    # Draw arc
    def draw_arc(screen, color, rect, start_angle, stop_angle, width):
        n_points = 100
        pointlist = []

        for i in range(n_points + 1):
            angle = math.radians(start_angle + (stop_angle - start_angle) * i / n_points)
            x = rect[0] + rect[2] // 2 + int(rect[2] // 2 * math.cos(angle))
            y = rect[1] + rect[3] // 2 - int(rect[3] // 2 * math.sin(angle))
            pointlist.append((x, y))

        pygame.draw.lines(screen, color, False, pointlist, width)
    # Draw online instrumentation
    def draw_online_instrumentation(self,screen, center_pos, label):
        # Create font and render text
        font = pygame.font.Font(None, 24)
        text = font.render(label, True, BLACK)
        text_width, text_height = text.get_size()

        # Calculate the size of the pill shape based on the text size
        padding = 10
        pill_width = text_width + 2 * padding
        pill_height = text_height + 2 * padding

        # Draw pill shape
        rect = (center_pos[0] - pill_width // 2, center_pos[1] - pill_height // 2, pill_width, pill_height)
        pygame.draw.rect(screen, BLACK, rect, 2)
        draw_arc(screen, BLACK, rect, 0, 180, 2)
        draw_arc(screen, BLACK, rect, 180, 360, 2)

        # Draw label
        text_rect = text.get_rect(center=center_pos)
        screen.blit(text, text_rect)
    # Draw a distillation column
    def draw_distillation_column(self,screen, center_pos, label, column_width, column_height, tray_thickness=2):
        # Create font and render text
        font = pygame.font.Font(None, 24)
        text = font.render(label, True, BLACK)
        text_width, text_height = text.get_size()

        # Calculate the size of the column based on the text size
        padding = 1

        # Calculate the number of trays and tray width based on column dimensions
        num_trays = int((column_height - 2 * padding) / (text_width + 2 * padding))
        tray_width = int(column_width / 2)

        # Draw column
        column_rect = pygame.Rect(center_pos[0] - column_width / 2, center_pos[1] - column_height / 2,
                                column_width, column_height)
        pygame.draw.rect(screen, BLACK, column_rect, 5)
        pygame.draw.rect(screen, WHITE, column_rect.inflate(-3, -3))

        # Draw trays
        tray_spacing = (column_height - 2 * padding) / (num_trays + 1)
        for i in range(num_trays):
            y = center_pos[1] - column_height / 2 + padding + (i + 1) * tray_spacing
            if i % 2 == 0:  # left side
                pygame.draw.line(screen, BLACK, (center_pos[0] - column_width / 2, y),
                                (center_pos[0] - column_width / 2 + tray_width, y), tray_thickness)
            else:  # right side
                pygame.draw.line(screen, BLACK, (center_pos[0] + column_width / 2, y),
                                (center_pos[0] + column_width / 2 - tray_width, y), tray_thickness)

        # Draw label
        rotated_text = pygame.transform.rotate(text, 90)
        rotated_text_rect = text.get_rect(center=center_pos)
        screen.blit(text, rotated_text_rect)
    # Draw a shell and tube heat exchanger
    def draw_shell_and_tube_heat_exchanger(self,screen, center_pos):
        # Draw shell
        pygame.draw.ellipse(screen, BLACK, (center_pos[0] - 50, center_pos[1] - 20, 100, 40), 2)

        # Draw tube bundle
        for i in range(5):
            pygame.draw.line(screen, BLACK, (center_pos[0] - 40 + i * 20, center_pos[1] - 15), (center_pos[0] - 40 + i * 20, center_pos[1] + 15), 2)

        # Draw channel heads
        pygame.draw.rect(screen, BLACK, (center_pos[0] - 60, center_pos[1] - 20, 10, 40), 2)
        pygame.draw.rect(screen, BLACK, (center_pos[0] + 50, center_pos[1] - 20, 10, 40), 2)
    # Draw a heat exchanger
    def draw_heat_exchanger(self,screen, center_pos, radius=20):
        # Draw circle
        pygame.draw.circle(screen, BLACK, center_pos, radius, 2)

        # Calculate the scale factor
        scale_factor = radius / 20

        # Draw zigzag line
        zigzag_points = [
            (center_pos[0] - radius, center_pos[1]),
            (center_pos[0] - radius + 5 * scale_factor, center_pos[1]),
            (center_pos[0] - 10 * scale_factor, center_pos[1] - 10 * scale_factor),
            (center_pos[0], center_pos[1] + 10 * scale_factor),
            (center_pos[0] + 10 * scale_factor, center_pos[1] - 10 * scale_factor),
            (center_pos[0] + radius - 5 * scale_factor, center_pos[1]),
            (center_pos[0] + radius, center_pos[1])
        ]
        pygame.draw.lines(screen, BLACK, False, zigzag_points, 2)
    # Draw a filter press
    def draw_filter_press(self,screen, center_pos, label, column_width, column_height, tray_thickness=2):
        # Create font and render text
        font = pygame.font.Font(None, 24)
        text = font.render(label, True, BLACK)
        text_width, text_height = text.get_size()

        # Draw column
        column_rect = pygame.Rect(center_pos[0] - column_width / 2, center_pos[1] - column_height / 2,
                                column_width, column_height)
        pygame.draw.rect(screen, BLACK, column_rect, 5)
        pygame.draw.rect(screen, WHITE, column_rect.inflate(-3, -3))

        # Calculate the number of trays based on column_height and text_height
        padding = 1
        num_trays = (column_height - text_width - 4 * padding) // (2 * padding)

        # Draw trays
        tray_spacing = (column_height - 2 * padding) / (num_trays + 1)
        for i in range(num_trays):
            y = center_pos[1] - column_height / 2 + padding + (i + 1) * tray_spacing
            pygame.draw.line(screen, BLACK, (center_pos[0] - column_width / 2, y),
                            (center_pos[0] + column_width / 2, y), tray_thickness)

        # Draw label
        rotated_text = pygame.transform.rotate(text, 90)
        rotated_text_rect = rotated_text.get_rect(center=center_pos)
        screen.blit(rotated_text, rotated_text_rect)
    # Draw a dome roof tank
    def draw_dome_roof_tank(self,screen, center_pos, width, height, color=BLACK):
        # Draw the tank body (rectangle)
        tank_rect = pygame.Rect(center_pos[0] - width / 2, center_pos[1], width, height)
        pygame.draw.rect(screen, color, tank_rect, 2)

        # Draw the dome (half-circle)
        dome_rect = pygame.Rect(center_pos[0] - width / 2, center_pos[1] - height / 2, width, height)
        pygame.draw.arc(screen, color, dome_rect, 0, math.pi, 2)
    # Draw a centrifugal pump
    def draw_centrifugal_pump(self,screen, center_pos, radius=20):
        pygame.draw.circle(screen, BLACK, center_pos, radius, 2)

        start_angle1 = math.pi / 2
        start_angle2 = 3 * math.pi / 2
        end_angle = 2 * math.pi

        start_point1 = (int(center_pos[0] + radius * math.cos(start_angle1)),
                        int(center_pos[1] - radius * math.sin(start_angle1)))
        start_point2 = (int(center_pos[0] + radius * math.cos(start_angle2)),
                        int(center_pos[1] - radius * math.sin(start_angle2)))
        end_point = (int(center_pos[0] + radius * math.cos(end_angle)),
                    int(center_pos[1] - radius * math.sin(end_angle)))

        pygame.draw.line(screen, BLACK, start_point1, end_point, 2)
        pygame.draw.line(screen, BLACK, start_point2, end_point, 2)
    # Draw a tank
    def draw_tank(screen, center_pos):
        pygame.draw.rect(screen, BLACK, (center_pos[0] - 25, center_pos[1] - 50, 50, 100), 2)
    # Draw a storage tank
    def draw_storage_tank(self, screen, x, y, width, height, border_width):
        # Draw the tank body
        tank_color = (255, 255, 255)  # White
        pygame.draw.rect(screen, tank_color, pygame.Rect(x, y, width, height // 2))
        # Draw the Liquid in the tank
        tank_color = (0, 0, 255)  # Blue
        pygame.draw.rect(screen, tank_color, pygame.Rect(x, y + height // 2, width, height // 2))
        tank_border_color = (0, 0, 0)  # Black
        pygame.draw.rect(screen, tank_border_color, pygame.Rect(x - border_width, y - border_width, width + 2 * border_width, height + 2 * border_width), border_width * 2)

    # Draw a right swing check valve
    def draw_left_swing_check_valve(screen, center_pos, triangle_base, color=BLACK, fill_color=WHITE):
        # Calculate triangle height
        triangle_height = (math.sqrt(3) / 2) * triangle_base

        # Calculate the coordinates of the four points
        left_triangle_base_top = (center_pos[0] - triangle_base / 2, center_pos[1] - triangle_height / 2)
        left_triangle_base_bottom = (center_pos[0] - triangle_base / 2, center_pos[1] + triangle_height / 2)
        right_triangle_base_top = (center_pos[0] + triangle_base / 2, center_pos[1] - triangle_height / 2)
        right_triangle_base_bottom = (center_pos[0] + triangle_base / 2, center_pos[1] + triangle_height / 2)
        touching_point = (center_pos[0], center_pos[1])

        # Draw the left triangle filled with white
        pygame.draw.polygon(screen, color, [left_triangle_base_top, left_triangle_base_bottom, touching_point])

        # Draw the outline of the left triangle
        pygame.draw.polygon(screen, color, [left_triangle_base_top, left_triangle_base_bottom, touching_point], 2)

        # Draw the right triangle filled with white
        pygame.draw.polygon(screen, fill_color, [right_triangle_base_top, right_triangle_base_bottom, touching_point])

        # Draw the outline of the right triangle
        pygame.draw.polygon(screen, color, [right_triangle_base_top, right_triangle_base_bottom, touching_point], 2)
    # Draw a right swing check valve
    def draw_right_swing_check_valve(screen, center_pos, triangle_base, color=BLACK, fill_color=WHITE):
        # Calculate triangle height
        triangle_height = (math.sqrt(3) / 2) * triangle_base

        # Calculate the coordinates of the four points
        left_triangle_base_top = (center_pos[0] - triangle_base / 2, center_pos[1] - triangle_height / 2)
        left_triangle_base_bottom = (center_pos[0] - triangle_base / 2, center_pos[1] + triangle_height / 2)
        right_triangle_base_top = (center_pos[0] + triangle_base / 2, center_pos[1] - triangle_height / 2)
        right_triangle_base_bottom = (center_pos[0] + triangle_base / 2, center_pos[1] + triangle_height / 2)
        touching_point = (center_pos[0], center_pos[1])

        # Draw the left triangle filled with white
        pygame.draw.polygon(screen, fill_color, [left_triangle_base_top, left_triangle_base_bottom, touching_point])

        # Draw the outline of the left triangle
        pygame.draw.polygon(screen, color, [left_triangle_base_top, left_triangle_base_bottom, touching_point], 2)

        # Draw the right triangle filled with white
        pygame.draw.polygon(screen, color, [right_triangle_base_top, right_triangle_base_bottom, touching_point])

        # Draw the outline of the right triangle
        pygame.draw.polygon(screen, color, [right_triangle_base_top, right_triangle_base_bottom, touching_point], 2)

    def draw_relief_valve(screen, center_pos):
        pygame.draw.polygon(screen, BLACK, [(center_pos[0] - 20, center_pos[1] + 20), (center_pos[0], center_pos[1] - 20), (center_pos[0] + 20, center_pos[1] + 20)], 2)
        pygame.draw.rect(screen, BLACK, (center_pos[0] - 5, center_pos[1] + 20, 10, 20), 2)

    def draw_flow_control_valve(screen, center_pos):
        pygame.draw.rect(screen, BLACK, (center_pos[0] - 25, center_pos[1] - 5, 50, 10), 2)
        pygame.draw.polygon(screen, BLACK, [(center_pos[0], center_pos[1] - 10), (center_pos[0] - 10, center_pos[1] - 20), (center_pos[0] + 10, center_pos[1] - 20)], 2)

    def draw_pipe_with_arrow(screen, start_pos, end_pos, label, label_above=True):
        # Draw the line
        pygame.draw.line(screen, BLACK, start_pos, end_pos, 2)

        # Calculate the arrowhead points
        arrow_length = 10
        angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
        arrow_angle = math.radians(30)

        point_a = (end_pos[0] - arrow_length * math.cos(angle + arrow_angle),
                end_pos[1] - arrow_length * math.sin(angle + arrow_angle))
        point_b = (end_pos[0] - arrow_length * math.cos(angle - arrow_angle),
                end_pos[1] - arrow_length * math.sin(angle - arrow_angle))

        # Draw arrowhead
        pygame.draw.polygon(screen, BLACK, [end_pos, point_a, point_b], 0)

        # Draw label
        font = pygame.font.Font(None, 24)
        text = font.render(label, True, BLACK)
        text_center_x = (start_pos[0] + end_pos[0]) // 2
        text_center_y = (start_pos[1] + end_pos[1]) // 2

        # Calculate label offset
        label_offset = 15
        offset_x = label_offset * math.sin(angle)
        offset_y = -label_offset * math.cos(angle)

        if not label_above:
            offset_x = -offset_x
            offset_y = -offset_y

        text_rect = text.get_rect(center=(text_center_x + offset_x, text_center_y + offset_y))
        
        # Rotate text
        rotated_text = pygame.transform.rotate(text, math.degrees(-angle))
        rotated_text_rect = rotated_text.get_rect(center=text_rect.center)

        screen.blit(rotated_text, rotated_text_rect)

    def draw_pipe_with_double_arrow(screen, start_pos, end_pos, label, label_above=True):
        # Draw the line
        pygame.draw.line(screen, BLACK, start_pos, end_pos, 2)

        # Calculate the arrowhead points
        arrow_length = 10
        angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
        arrow_angle = math.radians(30)

        def draw_arrowhead(end_pos, reverse=False):
            if reverse:
                angle_adjusted = angle + math.pi
            else:
                angle_adjusted = angle

            point_a = (end_pos[0] - arrow_length * math.cos(angle_adjusted + arrow_angle),
                    end_pos[1] - arrow_length * math.sin(angle_adjusted + arrow_angle))
            point_b = (end_pos[0] - arrow_length * math.cos(angle_adjusted - arrow_angle),
                    end_pos[1] - arrow_length * math.sin(angle_adjusted - arrow_angle))
            pygame.draw.polygon(screen, BLACK, [end_pos, point_a, point_b], 0)

        # Draw arrowheads at both ends
        draw_arrowhead(end_pos)
        draw_arrowhead(start_pos, reverse=True)

        # Draw label
        font = pygame.font.Font(None, 24)
        text = font.render(label, True, BLACK)
        text_center_x = (start_pos[0] + end_pos[0]) // 2
        text_center_y = (start_pos[1] + end_pos[1]) // 2

        # Calculate label offset
        label_offset = 15
        offset_x = label_offset * math.sin(angle)
        offset_y = -label_offset * math.cos(angle)

        if not label_above:
            offset_x = -offset_x
            offset_y = -offset_y

        text_rect = text.get_rect(center=(text_center_x + offset_x, text_center_y + offset_y))

        # Rotate text
        rotated_text = pygame.transform.rotate(text, math.degrees(-angle))
        rotated_text_rect = rotated_text.get_rect(center=text_rect.center)

        screen.blit(rotated_text, rotated_text_rect)

    def draw_dashed_line(screen, start_pos, end_pos, color, dash_length=5, gap_length=5):
        x1, y1 = start_pos
        x2, y2 = end_pos
        dx = x2 - x1
        dy = y2 - y1
        distance = math.sqrt(dx * dx + dy * dy)
        dashes = int(distance / (dash_length + gap_length))
        dash_step = dx / (dashes * 2)
        gap_step = dy / (dashes * 2)

        for i in range(dashes):
            x_dash_start = x1 + (i * 2 * (dash_step + gap_step))
            y_dash_start = y1 + (i * 2 * (dash_step + gap_step))
            x_dash_end = x_dash_start + dash_step
            y_dash_end = y_dash_start + gap_step
            pygame.draw.line(screen, color, (x_dash_start, y_dash_start), (x_dash_end, y_dash_end), 2)

    def draw_digital_signal(screen, start_pos, end_pos, dash_length=10, color=BLACK):
        length = math.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        num_dashes = int(length / (dash_length * 2))

        for i in range(num_dashes):
            start = (
                start_pos[0] + (i * 2 * dash_length / length) * (end_pos[0] - start_pos[0]),
                start_pos[1] + (i * 2 * dash_length / length) * (end_pos[1] - start_pos[1])
            )
            end = (
                start_pos[0] + ((2 * i + 1) * dash_length / length) * (end_pos[0] - start_pos[0]),
                start_pos[1] + ((2 * i + 1) * dash_length / length) * (end_pos[1] - start_pos[1])
            )
            pygame.draw.line(screen, color, start, end, 2)

    def draw_pneumatic_line(screen, start_pos, end_pos, dash_length=20, color=BLACK):
        length = math.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        num_dashes = int(length / (dash_length * 2))
        angle = math.atan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])
        diagonal_angle = angle + math.pi / 4
        
        # Draw the main solid line
        pygame.draw.line(screen, color, start_pos, end_pos, 2)

        for i in range(num_dashes):
            dash_start_fraction = (i * 2 * dash_length / length)
            dash_end_fraction = ((2 * i + 1) * dash_length / length)

            center = (
                start_pos[0] + dash_start_fraction * (end_pos[0] - start_pos[0]),
                start_pos[1] + dash_start_fraction * (end_pos[1] - start_pos[1])
            )
            start = (
                center[0] - (dash_length / 4) * math.cos(diagonal_angle),
                center[1] - (dash_length / 4) * math.sin(diagonal_angle)
            )
            end = (
                center[0] + (dash_length / 4) * math.cos(diagonal_angle),
                center[1] + (dash_length / 4) * math.sin(diagonal_angle)
            )
            pygame.draw.line(screen, color, start, end, 2)
    
    def draw_instrumentation(screen, center_pos, symbol_type):
        pygame.draw.circle(screen, BLACK, center_pos, 20, 2)

        font = pygame.font.Font(None, 24)
        text = font.render(symbol_type, True, BLACK)
        text_rect = text.get_rect(center=center_pos)
        screen.blit(text, text_rect)

    def draw_vertical_drum(screen, center_pos, width, height, label, arc_height_ratio=2, color=BLACK):
        # Create font and render text
        font = pygame.font.Font(None, 24)
        text = font.render(label, True, BLACK)

        # Calculate the height of the enclosing rectangle for the arcs
        arc_height = height / arc_height_ratio

        # Draw the top dome (arc)
        top_dome_rect = pygame.Rect(center_pos[0] - width / 2, center_pos[1] - height / 2 - arc_height/2, width, arc_height)
        pygame.draw.arc(screen, color, top_dome_rect, 0, math.pi, 2)

        # Draw the bottom dome (arc)
        bottom_dome_rect = pygame.Rect(center_pos[0] - width / 2, center_pos[1] + height / 2 - arc_height/2, width, arc_height)
        pygame.draw.arc(screen, color, bottom_dome_rect, math.pi, 2 * math.pi, 2)

        # Calculate tangent points for lines
        top_left_tangent = (center_pos[0] - width / 2, center_pos[1] - height / 2)
        bottom_left_tangent = (center_pos[0] - width / 2, center_pos[1] + height / 2)
        top_right_tangent = (center_pos[0] + width / 2, center_pos[1] - height / 2)
        bottom_right_tangent = (center_pos[0] + width / 2, center_pos[1] + height / 2)

        # Connect the points with lines
        pygame.draw.line(screen, color, top_left_tangent, bottom_left_tangent, 2)
        pygame.draw.line(screen, color, top_right_tangent, bottom_right_tangent, 2)

        # Draw label
        text_rect = text.get_rect(center=center_pos)
        screen.blit(text, text_rect)

    def draw_venturi(screen, center_pos, width, height, arc_height_ratio=1.75, color=BLACK):
        # Calculate the width of the enclosing rectangle for the arcs
        arc_width = width / arc_height_ratio

        # Draw the left dome (arc)
        left_dome_rect = pygame.Rect(center_pos[0] - width / 2 - arc_width / 2, center_pos[1] - height / 2, arc_width, height)
        pygame.draw.arc(screen, color, left_dome_rect, -math.pi / 2, math.pi / 2, 2)

        # Draw the right dome (arc)
        right_dome_rect = pygame.Rect(center_pos[0] + width / 2 - arc_width / 2, center_pos[1] - height / 2, arc_width, height)
        pygame.draw.arc(screen, color, right_dome_rect, math.pi / 2, 3 * math.pi / 2, 2)

        # Calculate tangent points for lines
        top_left_tangent = (center_pos[0] - width / 2, center_pos[1] - height / 2)
        top_right_tangent = (center_pos[0] + width / 2, center_pos[1] - height / 2)
        bottom_left_tangent = (center_pos[0] - width / 2, center_pos[1] + height / 2)
        bottom_right_tangent = (center_pos[0] + width / 2, center_pos[1] + height / 2)

        # Connect the points with lines
        pygame.draw.line(screen, color, top_left_tangent, top_right_tangent, 2)
        pygame.draw.line(screen, color, bottom_left_tangent, bottom_right_tangent, 2)

    def draw_sine_wave(screen, center_pos, width, height, arc_height_ratio=2, color=BLACK):
        # Calculate the width of the enclosing rectangle for the arcs
        arc_width = width / arc_height_ratio

        # Draw the left dome (arc)
        left_dome_rect = pygame.Rect(center_pos[0] - width / 2, center_pos[1] - height / 2, arc_width, height)
        pygame.draw.arc(screen, color, left_dome_rect, 0, math.pi, 2)

        # Draw the right dome (arc)
        right_dome_rect = pygame.Rect(center_pos[0] + width / 2 - arc_width, center_pos[1] - height / 2, arc_width, height)
        pygame.draw.arc(screen, color, right_dome_rect, math.pi, 2 * math.pi, 2)

        # Calculate tangent points for lines
        top_left_tangent = (center_pos[0] - width / 2 + arc_width / 2, center_pos[1] - height / 2)
        top_right_tangent = (center_pos[0] + width / 2 - arc_width / 2, center_pos[1] - height / 2)
        bottom_left_tangent = (center_pos[0] - width / 2 + arc_width / 2, center_pos[1] + height / 2)
        bottom_right_tangent = (center_pos[0] + width / 2 - arc_width / 2, center_pos[1] + height / 2)

        # Connect the points with lines
        pygame.draw.line(screen, color, top_left_tangent, top_right_tangent, 2)
        pygame.draw.line(screen, color, bottom_left_tangent, bottom_right_tangent, 2)

    def draw_horizontal_drum(screen, center_pos, width, height, label, arc_height_ratio=2, color=BLACK):
        # Create font and render text
        font = pygame.font.Font(None, 24)
        text = font.render(label, True, BLACK)

        # Calculate the width of the enclosing rectangle for the arcs
        arc_width = width / arc_height_ratio

        # Draw the left dome (arc)
        left_dome_rect = pygame.Rect(center_pos[0] - width / 2, center_pos[1] - height / 2, arc_width, height)
        pygame.draw.arc(screen, color, left_dome_rect, math.pi / 2, 3 * math.pi / 2, 2)

        # Draw the right dome (arc)
        right_dome_rect = pygame.Rect(center_pos[0] + width / 2 - arc_width, center_pos[1] - height / 2, arc_width, height)
        pygame.draw.arc(screen, color, right_dome_rect, -math.pi / 2, math.pi / 2, 2)

        # Calculate tangent points for lines
        top_left_tangent = (center_pos[0] - width / 2 + arc_width / 2, center_pos[1] - height / 2)
        top_right_tangent = (center_pos[0] + width / 2 - arc_width / 2, center_pos[1] - height / 2)
        bottom_left_tangent = (center_pos[0] - width / 2 + arc_width / 2, center_pos[1] + height / 2)
        bottom_right_tangent = (center_pos[0] + width / 2 - arc_width / 2, center_pos[1] + height / 2)

        # Connect the points with lines
        pygame.draw.line(screen, color, top_left_tangent, top_right_tangent, 2)
        pygame.draw.line(screen, color, bottom_left_tangent, bottom_right_tangent, 2)

        # Draw label
        text_rect = text.get_rect(center=center_pos)
        screen.blit(text, text_rect)

    def draw_cstr_with_heating_jacket(screen, center_pos, width, height, color=BLACK):
        # Draw the outer rectangle for the CSTR
        outer_rect = pygame.Rect(center_pos[0] - width / 2, center_pos[1] - height / 2, width, height)
        pygame.draw.rect(screen, color, outer_rect, 2)

        # Draw the inner rectangle for the heating jacket
        inner_width = width * 0.8
        inner_height = height * 0.9
        inner_rect = pygame.Rect(center_pos[0] - inner_width / 2, center_pos[1] - inner_height / 2, inner_width, inner_height)
        pygame.draw.rect(screen, color, inner_rect, 2)

        # Draw the mixing paddle (vertical line)
        paddle_start = (center_pos[0], center_pos[1] - inner_height / 2)
        paddle_end = (center_pos[0], center_pos[1] + inner_height // 4)
        pygame.draw.line(screen, color, paddle_start, paddle_end, 2)

        # Draw the mixing paddle ellipses
        ellipse_width = inner_width // 3
        ellipse_height = inner_height // 10
        left_ellipse_rect = pygame.Rect(paddle_end[0] - ellipse_width, paddle_end[1] - ellipse_height / 2, ellipse_width, ellipse_height)
        right_ellipse_rect = pygame.Rect(paddle_end[0], paddle_end[1] - ellipse_height / 2, ellipse_width, ellipse_height)

        pygame.draw.ellipse(screen, color, left_ellipse_rect, 2)
        pygame.draw.ellipse(screen, color, right_ellipse_rect, 2)

        # Draw the motor box on top of the paddle
        motor_width = width / 4
        motor_height = height / 8
        motor_rect = pygame.Rect(center_pos[0] - motor_width / 2, paddle_start[1] - motor_height, motor_width, motor_height)
        pygame.draw.rect(screen, color, motor_rect, 2)

        # Draw motor fins
        num_fins = 4
        fin_spacing = motor_width / (num_fins + 1)
        for i in range(num_fins):
            fin_start_x = motor_rect.left + (i + 1) * fin_spacing
            fin_start = (fin_start_x, paddle_start[1] - motor_height)
            fin_end = (fin_start_x, paddle_start[1])
            pygame.draw.line(screen, color, fin_start, fin_end, 2)

    def draw_open_hand_valve(screen, center_pos, triangle_base, color=BLACK, fill_color=WHITE):
        # Calculate triangle height
        triangle_height = (math.sqrt(3) / 2) * triangle_base

        # Calculate the coordinates of the four points
        left_triangle_base_top = (center_pos[0] - triangle_base / 2, center_pos[1] - triangle_height / 2)
        left_triangle_base_bottom = (center_pos[0] - triangle_base / 2, center_pos[1] + triangle_height / 2)
        right_triangle_base_top = (center_pos[0] + triangle_base / 2, center_pos[1] - triangle_height / 2)
        right_triangle_base_bottom = (center_pos[0] + triangle_base / 2, center_pos[1] + triangle_height / 2)
        touching_point = (center_pos[0], center_pos[1])

        # Draw the left triangle filled with white
        pygame.draw.polygon(screen, fill_color, [left_triangle_base_top, left_triangle_base_bottom, touching_point])

        # Draw the outline of the left triangle
        pygame.draw.polygon(screen, color, [left_triangle_base_top, left_triangle_base_bottom, touching_point], 2)

        # Draw the right triangle filled with white
        pygame.draw.polygon(screen, fill_color, [right_triangle_base_top, right_triangle_base_bottom, touching_point])

        # Draw the outline of the right triangle
        pygame.draw.polygon(screen, color, [right_triangle_base_top, right_triangle_base_bottom, touching_point], 2)

    def draw_closed_hand_valve(screen, center_pos, triangle_base, color=BLACK):
        # Calculate triangle height
        triangle_height = (math.sqrt(3) / 2) * triangle_base

        # Calculate the coordinates of the four points
        left_triangle_base_top = (center_pos[0] - triangle_base / 2, center_pos[1] - triangle_height / 2)
        left_triangle_base_bottom = (center_pos[0] - triangle_base / 2, center_pos[1] + triangle_height / 2)
        right_triangle_base_top = (center_pos[0] + triangle_base / 2, center_pos[1] - triangle_height / 2)
        right_triangle_base_bottom = (center_pos[0] + triangle_base / 2, center_pos[1] + triangle_height / 2)
        touching_point = (center_pos[0], center_pos[1])

        # Draw the left triangle filled with black color
        pygame.draw.polygon(screen, color, [left_triangle_base_top, left_triangle_base_bottom, touching_point])

        # Draw the right triangle filled with black color
        pygame.draw.polygon(screen, color, [right_triangle_base_top, right_triangle_base_bottom, touching_point])

    def draw_open_on_off_valve(screen, center_pos, triangle_base, symbol_type, color=BLACK, fill_color=WHITE):
        # Calculate triangle height
        triangle_height = (math.sqrt(3) / 2) * triangle_base

        # Calculate the coordinates of the four points
        left_triangle_base_top = (center_pos[0] - triangle_base / 2, center_pos[1] - triangle_height / 2)
        left_triangle_base_bottom = (center_pos[0] - triangle_base / 2, center_pos[1] + triangle_height / 2)
        right_triangle_base_top = (center_pos[0] + triangle_base / 2, center_pos[1] - triangle_height / 2)
        right_triangle_base_bottom = (center_pos[0] + triangle_base / 2, center_pos[1] + triangle_height / 2)
        touching_point = (center_pos[0], center_pos[1])

        # Draw the left triangle filled with white
        pygame.draw.polygon(screen, fill_color, [left_triangle_base_top, left_triangle_base_bottom, touching_point])

        # Draw the outline of the left triangle
        pygame.draw.polygon(screen, color, [left_triangle_base_top, left_triangle_base_bottom, touching_point], 2)

        # Draw the right triangle filled with white
        pygame.draw.polygon(screen, fill_color, [right_triangle_base_top, right_triangle_base_bottom, touching_point])

        # Draw the outline of the right triangle
        pygame.draw.polygon(screen, color, [right_triangle_base_top, right_triangle_base_bottom, touching_point], 2)

        # Draw the short line connecting touching point to the instrumentation symbol
        line_end = (center_pos[0], center_pos[1] - triangle_height // 2 - 10)
        pygame.draw.line(screen, color, touching_point, line_end, 2)

        # Draw the instrumentation symbol
        instrument_center = (center_pos[0], center_pos[1] - triangle_height // 2 - triangle_base)
        draw_instrumentation(screen, instrument_center, symbol_type)

    def draw_closed_on_off_valve(screen, center_pos, triangle_base, symbol_type, color=BLACK):
        # Calculate triangle height
        triangle_height = (math.sqrt(3) / 2) * triangle_base

        # Calculate the coordinates of the four points
        left_triangle_base_top = (center_pos[0] - triangle_base / 2, center_pos[1] - triangle_height / 2)
        left_triangle_base_bottom = (center_pos[0] - triangle_base / 2, center_pos[1] + triangle_height / 2)
        right_triangle_base_top = (center_pos[0] + triangle_base / 2, center_pos[1] - triangle_height / 2)
        right_triangle_base_bottom = (center_pos[0] + triangle_base / 2, center_pos[1] + triangle_height / 2)
        touching_point = (center_pos[0], center_pos[1])

        # Draw the left triangle filled
        pygame.draw.polygon(screen, color, [left_triangle_base_top, left_triangle_base_bottom, touching_point])

        # Draw the right triangle filled
        pygame.draw.polygon(screen, color, [right_triangle_base_top, right_triangle_base_bottom, touching_point])

        # Draw the short line connecting touching point to the instrumentation symbol
        line_end = (center_pos[0], center_pos[1] - triangle_height // 2 - 10)
        pygame.draw.line(screen, color, touching_point, line_end, 2)

        # Draw the instrumentation symbol
        instrument_center = (center_pos[0], center_pos[1] - triangle_height // 2 - triangle_base)
        draw_instrumentation(screen, instrument_center, symbol_type)
    
    def render(self, screen):
        # Draw connections
        for connection in self.flowsheet.connections:
            # You might need to get the position and size of the connection elements
            # to draw them correctly, either by adding attributes or by using separate
            # position/size dictionaries
            source_pos = connection.source.position
            destination_pos = connection.destination.position

            # Choose the appropriate draw_* method depending on the type of connection
            self.draw_pipe_with_arrow(screen, source_pos, destination_pos)

        # Draw blocks
        for block in self.flowsheet.blocks:
            # You might need to get the position and size of the block to draw it correctly
            block_pos = block.position
            block_size = block.size

            # Choose the appropriate draw_* method depending on the type of block
            if isinstance(block, StorageTank):
                self.draw_storage_tank(screen, block_pos, block_size)
            # Add more conditions for other block types

        # Draw streams
        for stream in self.flowsheet.streams:
            # You might need to get the position and size of the stream to draw it correctly
            stream_pos = stream.position
            stream_size = stream.size

            # Choose the appropriate draw_* method depending on the type of stream
            self.draw_arrow_with_label(screen, stream_pos, stream_size, stream.name)

        # You can also draw other elements like labels, grid, or background here
# Define the Flowhsheet Class, This is where all the info will be saved
class Flowsheet:
    def __init__(self, name):
        self.name = name
        self.blocks = []
        self.connections = []
        self.streams = []

    def add_block(self, block):
        self.blocks.append(block)

    def add_connection(self, connection):
        self.connections.append(connection)

    def add_stream(self, stream):
        self.streams.append(stream)

    def print_info(self):
        print(f"Components in {self.name}:")
        for block in self.blocks:
            for input_stream in block.inputs:
                for component in input_stream.components:
                    print(f"- {component.name}")
            for output_stream in block.outputs:
                for component in output_stream.components:
                    print(f"- {component.name}")

        print(f"Streams in {self.name}:")
        for stream in self.streams:
            print(f"- {stream.name}")

        print(f"Blocks in {self.name}:")
        for block in self.blocks:
            print(f"- {block.name}")
# The Block is an abstract class that can take on the form of other equiptment, it take input and output
class Block:
    def __init__(self, name):
        self.name = name
        self.inputs = []
        self.outputs = []

    def add_input(self, input_stream):
        self.inputs.append(input_stream)

    def add_output(self, output_stream):
        self.outputs.append(output_stream)
# This class connects Streams and blocks
class Connection:
    def __init__(self, source, destination):
        self.source = source
        self.destination = destination
# The Stream class containes components which are connected to blocks via the connections
class Stream:
    def __init__(self, name, components):
        self.name = name
        self.components = components
# Components are the molecules are chemicals that make up streams
class Component:
    def __init__(self, name):
        self.name = name
# ---------------Block subclasses also equiptment---------------
# The StorageTank class is a subclass of Block
class Tank(Block):
    def __init__(self, name, position):
        super().__init__(name)
        self.position = position
        self.x, self.y = position
# The CentrifugalPump class is a subclass of Block
class CentrifugalPump(Block):
    def __init__(self, name, position):
        super().__init__(name)
        self.position = position
        self.x, self.y = position
# The Hand Valve class is a subclass of Block
class HandValve(Block):
    def __init__(self, name, position):
        super().__init__(name)
        self.position = position
        self.x, self.y = position
# The ReliefValve class is a subclass of Block
class ReliefValve(Block):
    def __init__(self, name, position):
        super().__init__(name)
        self.position = position
        self.x, self.y = position
# The ControlValve class is a subclass of Block
class ControlValve(Block):
    def __init__(self, name, position):
        super().__init__(name)
        self.position = position
        self.x, self.y = position
# The FlashTank class is a subclass of Block
class FlashTank(Block):
    def __init__(self, name, position):
        super().__init__(name)
        self.position = position
        self.x, self.y = position
# The DistillationColumn class is a subclass of Block
class DistillationColumn(Block):
    def __init__(self, name, position):
        super().__init__(name)
        self.position = position
        self.x, self.y = position
# The StrippingColumn class is a subclass of Block
class StrippingColumn(Block):
    def __init__(self, name, position):
        super().__init__(name)
        self.position = position
        self.x, self.y = position
# The AbsorptionColumn class is a subclass of Block
class AbsorptionColumn(Block):
    def __init__(self, name, position):
        super().__init__(name)
        self.position = position
        self.x, self.y = position
# The Pipe class is a subclass of Block
class Pipe(Block):
    def __init__(self, name, position):
        super().__init__(name)
        self.position = position
        self.x, self.y = position
# The GeneralHeatExchanger class is a subclass of Block
class GeneralHeatExchanger(Block):
    def __init__(self, name, position):
        super().__init__(name)
        self.position = position
        self.x, self.y = position
# The STHeatExchanger class is a subclass of Block
class STHeatExchanger(Block):
    def __init__(self, name, position):
        super().__init__(name)
        self.position = position
        self.x, self.y = position
# The PlateHeatExchanger class is a subclass of Block
class PlateHeatExchanger(Block):
    def __init__(self, name, position):
        super().__init__(name)
        self.position = position
        self.x, self.y = position
# Enums for block types
class BlockType(Enum):
    Tank = "Tank"
    CentrifugalPump = "Centrifugal Pump"
    HandValve = "Hand Valve"
    ControlValve = "Control Valve"
    ReliefValve = "Relief Valve"
    FlashTank = "Flash Tank"
    DistillationColumn = "Distillation Column"  # Add this line
    StrippingColumn = "Stripping Column"
    AbsorptionColumn = "Absorption Column"
    Pipe = "Pipe"
    GeneralHeatExchanger = "Heat Exchanger"
    STHeatExchanger = "Shell and Tube Heat Exchanger"
    PlateHeatExchanger = "Plate Heat Exchanger"
    CSTR = "Continuous Stirred Tank Reactor"
    PFR = "Plug Flow Reactor"
    BatchReactor = "Batch Reactor"

# Create the MainMenuPage object
main_menu_page = MainMenuPage()

# Create the PageManager object with MainMenuPage as the initial page
page_manager = PageManager(initial_page=main_menu_page)

clock = pygame.time.Clock()
# Main game loop
while page_manager.running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            page_manager.running = False
            pygame.quit()
        else:
            # Pass the event to the current page
            page_manager.handle_event(event)

    # Check if a new flowsheet has been created
    if isinstance(page_manager.current_page, NewFlowsheetPage) and page_manager.current_page.new_flowsheet_created:
        filename = page_manager.current_page.text_input.strip() + ".pkl"
        print("Flowsheet created: ", filename)
        flowsheet_sim = RunFlowsheetSimulation(filename, screen, page_manager)
        flowsheet_sim.run()

        if flowsheet_sim.go_to_main_menu:
            page_manager.current_page = MainMenuPage()

    # Render the current page
    page_manager.render(screen)
    pygame.display.flip()
    clock.tick(60)

print("Thanks for using Franks Chemical Simulator")
