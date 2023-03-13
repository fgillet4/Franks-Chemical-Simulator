import pygame
import csv
import pickle
flowsheet_version = "Flowsheet Simulator v.1.0.0"
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128, 128)
# Set up Menu Layers with True False Switching

#Main Menu Layer
in_main_menu = True
# Initialize Pygame
pygame.init()

# Set up the Pygame window
WINDOW_SIZE = (1600, 800)
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Frank's Chemical Process Simulator")

# Set up fonts
font = pygame.font.Font(None, 40)

# Define buttons
button_width = 300
button_height = 75
button_padding = 50
button_start_x = (WINDOW_SIZE[0] - button_width) / 2
button_start_y = (WINDOW_SIZE[1] - (button_height + button_padding) * 5) / 2

# Defines the abstract class called page
class Page:
    """an abstract class representing a single screen"""

    def render(self, pygame_screen):
        pass
# Defines the Page Manager
class PageManager:
    def __init__(self, initial_page):
        self.current_page = initial_page
        self.current_page.manager = self  # set the manager attribute of the initial page

    def go_to(self, page):
        self.current_page = page
        self.current_page.manager = self  # set the manager attribute of the new page

    def handle_event(self, event):
        self.current_page.handle_event(event)

    def render(self, screen):
        self.current_page.render(screen)
# Defines the Main Menu pages
class MainMenuPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x-15, button_start_y, button_width+30, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x-10, button_start_y + (button_height + button_padding) * 2, button_width+20, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 4, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 5, button_width, button_height)
    ]
        self.menu_texts = [
        "Flowsheet Simulation",
        "Equipment Sizing",
        "Process Economics",
        "Process Safety",
        "Physical Properties",
        "Quit"
    ]
        self.in_flowsheet_sim = False
        self.in_equiptment_sizing = False
        self.in_capital_cost = False
        self.in_process_safety = False
        self.in_physical_properties = False
        self.in_quit = False
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
                        self.in_quit = True
        if self.in_flowsheet_sim:
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_equiptment_sizing:
            self.manager.go_to(EquipmentSizingPage())
        elif self.in_capital_cost:
            self.manager.go_to(CapitalCostEstimationPage())
        elif self.in_process_safety:
            self.manager.go_to(ProcessSafetyPage())
        elif self.in_physical_properties:
            self.manager.go_to(PhysicalPropertiesPage())
        elif self.in_quit:
            pygame.quit()
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        for i, rect in enumerate(self.menu_rects):
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
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
            self.manager.go_to(NewFlowsheetPage())
        elif self.in_load_flowsheet:
            self.manager.go_to(LoadFlowsheetPage())
        elif self.in_back:
            self.manager.go_to(MainMenuPage())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render(flowsheet_version, True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        for i, rect in enumerate(self.menu_rects):
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            pygame_screen.blit(text, text_rect)
# Define the New Flowsheet Page
class NewFlowsheetPage(Page):
    def __init__(self, page_manager=None):
        self.input_rect = pygame.Rect(100, 300, 400, 50)
        self.input_text = ""
        self.back_rect = pygame.Rect(50, 400, 100, 50)  # Back button rectangle
        self.back_text = "Back"
        self.enter_rect = pygame.Rect(self.input_rect.right + 50, self.input_rect.top, 100, self.input_rect.height)  # Enter button rectangle
        self.enter_text = "Enter"
        self.in_simulation = False
        self.in_back = False
        self.in_enter = False

    def handle_event(self, event):
        # Handle keyboard events
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                # Start the simulation with the entered name
                filename = self.input_text.strip() + ".pkl"
                print(filename)
                start_simulation(filename)
                self.in_simulation = True
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
            elif self.enter_rect.collidepoint(event.pos):
                self.in_enter = True

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
        screen.blit(title, (100, 100))

        input_title = font.render("Type the name of the flowsheet and hit Enter:", True, BLACK)
        screen.blit(input_title, (100, 200))

        pygame.draw.rect(screen, WHITE, self.input_rect)
        self.update_input_field()  # Call the update_input_field() method here
        pygame.draw.rect(screen, BLACK, self.input_rect, 2)

        enter_button = font.render(self.enter_text, True, BLACK)
        enter_rect = enter_button.get_rect()
        enter_rect.center = self.enter_rect.center
        screen.blit(enter_button, enter_rect)

        # Draw the back button
        pygame.draw.rect(screen, GRAY, self.back_rect)
        back_button = font.render(self.back_text, True, BLACK)
        back_button_rect = back_button.get_rect()
        back_button_rect.center = self.back_rect.center
        screen.blit(back_button, back_button_rect)

        pygame.display.update(self.input_rect)

        # Check if the "Back" or "Enter" button has been pressed
        if self.in_back:
            self.in_back = False
            self.manager.go_to(FlowsheetSimulationPage())
        elif self.in_enter:
            self.in_enter = False
            filename = self.input_text.strip() + ".pkl"
            print(filename)
            start_simulation(filename)
            self.in_simulation = True
# Define Load Flowsheet Page
class LoadFlowsheetPage(Page):
    pass
# Define Equiptment Sizing Page
class EquipmentSizingPage(Page):
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
        "Heat Exchangers",
        "Separations",
        "Pumps",
        "Reactors",
        "Controls",
        "Back"
    ]
        self.in_heat_exchangers= False
        self.in_separations = False
        self.in_pumps = False
        self.in_reactos = False
        self.in_controls = False
        self.in_back = False
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
        "Finned Tube",
        "Double Pipe",
        "Back"
    ]
        self.in_shell_tube_heat_exchanger= False
        self.in_plate_heat_exchanger = False
        self.in_spiral_heat_exchanger = False
        self.in_finned_tube_heat_exchanger = False
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
                        self.in_finned_tube_heat_exchanger = True
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
        elif self.in_finned_tube_heat_exchanger:
            self.manager.go_to(FinnedHexPage())
        elif self.in_double_pipe_heat_exchanger:
            self.manager.go_to(DoublePipeHexPage())
        elif self.in_back:
            self.manager.go_to(EquipmentSizingPage())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Type of Heat Exchanger", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        for i, rect in enumerate(self.menu_rects):
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            pygame_screen.blit(text, text_rect)
# Define Shell and Tube Hex Page
class ShellTubeHexPage(Page):
    pass
# Define Plate Heat Exchanger Page
class PlateHexPage(Page):
    pass
# Define Spiral Heat Exchanger Page
class SpiralHexPage(Page):
    pass
# Define Finned Heat Exchanger Page
class FinnedHexPage(Page):
    pass
# Define Double Pipe Heat Exchanger Page
class DoublePipeHexPage(Page):
    pass
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
        text = font.render("Select Separations Type", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        for i, rect in enumerate(self.menu_rects):
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            pygame_screen.blit(text, text_rect)
# Define The Gas Separations Page
class GasSeparationsPage(Page):
    pass
# Define The Liquid Separations Page
class LiquidSeparationsPage(Page):
    pass
# Define The Solids Separations Page
class SolidSeparationsPage(Page):
    pass
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
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
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
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Multi-Stage Centrifugal Pumps Page
class MultiStagePumpsPage(Page):
    pass
# Define the single stage centrifugal pumps page
class SingleStagePumpsPage(Page):
    pass
# Define the Magnetic Pumps Page
class MagneticDrivePumpsPage(Page):
    pass
# Define the Submersible Pumps Page
class SubmersiblePumpsPage(Page):
    pass
# Define Positive Displacement Pumps Page
class PositiveDisplacementPumpsPage(Page):
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
        "Piston",
        "Rotary lobe",
        "Diaphragm",
        "Peristaltic",
        "Screw",
        "Back"
    ]
        self.in_heat_exchangers= False
        self.in_separations = False
        self.in_pumps = False
        self.in_reactos = False
        self.in_controls = False
        self.in_back = False
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
            self.manager.go_to(PumpsPage())
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
# Define Ejector Pumps Page
class EjectorPumpsPage(Page):
    pass
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
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
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
        text = font.render("Types of Controls", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        for i, rect in enumerate(self.menu_rects):
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            pygame_screen.blit(text, text_rect)
# Define Valves Page
class ValvesPage(Page):
    pass
# Define Transfer Functions Page
class TransferFnPage(Page):
    pass
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
    pass
# Bends Page
class BendsPage(Page):
    pass
# Define the Orifices Page
class OrificePage(Page):
    pass
# Define the Tees Page
class TeesPage(Page):
    pass
# Define the Vena Contracta Page
class VenaContractaPage(Page):
    pass
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
    pass
# Define Pressure Sensors Page
class PressureSensorsPage(Page):
    pass
# Define the Flow Sensors Page
class FlowSensorsPage(Page):
    pass
# Define the Level Sensors Page
class LevelSensorsPage(Page):
    pass
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
    pass
# Define the Conductivity Sensors Page
class ConductivitySensorsPage(Page):
    pass
# Define FT-IR Sensors Page
class FtIrPage(Page):
    pass
# Define Gas Alarms Page
class GasAlarmsPage(Page):
    pass
# Define Capital Cost Page
class CapitalCostEstimationPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x-100, button_start_y, button_width+200, button_height+20),
        pygame.Rect(button_start_x-125, button_start_y + button_height + button_padding, button_width+250, button_height+20),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height)
    ]
        self.menu_texts = [
        "Estimate Flowsheet Capital Cost",
        "Edit Capital Cost Estimation Factors",
        "Back"
    ]
        self.in_estimate_flowsheet_capital_cost= False
        self.in_edit_capital_cost_estimation_factors = False
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
                        self.in_back = True
        if self.in_estimate_flowsheet_capital_cost:
            self.manager.go_to(EstimateFlowsheetCapitalCostEstimationPage())
        elif self.in_edit_capital_cost_estimation_factors:
            self.manager.go_to(EditCapitalCostEstimationParametersPage())
        elif self.in_back:
            self.manager.go_to(MainMenuPage())
    def render(self, pygame_screen):
        # Draw the main menu
        screen.fill(WHITE)
        text = font.render("Capital Cost Estimation", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        for i, rect in enumerate(self.menu_rects):
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            pygame_screen.blit(text, text_rect)
# Define The Estimate 
class EstimateFlowsheetCapitalCostEstimationPage(Page):
    pass
# Define Edit Capital Cost Estimation Parameters Page
class EditCapitalCostEstimationParametersPage(Page):
    pass
# Define Process Safety Page
class ProcessSafetyPage(Page):
    def __init__(self,page_manager = None):
        self.menu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
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
        text = font.render("Estimate Flowsheet Safety", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        for i, rect in enumerate(self.menu_rects):
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            pygame_screen.blit(text, text_rect)
# Define Flowsheet Safety Estimation Page
class EstimateFlowsheetSafetyPage:
    pass
# Define Chemical Safety Page
class FindChemicalSafetyPage:
    pass
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
        "Releif Valves",
        "Gas Alarms",
        "Back"
    ]
        self.in_rupture_disk= False
        self.in_releif_valves = False
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
                        self.in_releif_valves = True
                    elif i == 2:
                        self.in_gas_alarms = True
                    elif i == 3:
                        self.in_back = True
        if self.in_rupture_disk:
            self.manager.go_to(RuptureDiskPage())
        elif self.in_releif_valves:
            self.manager.go_to(ReleifValvesPage())
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
class RuptureDiskPage:
    pass
# Define Releif Valves Page
class ReleifValvesPage:
    pass
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
            self.manager.go_to(AddChemicalPage())
        elif self.in_find_chemical:
            self.manager.go_to(FindChemicalPage())
        elif self.in_back:
            self.manager.go_to(MainMenuPage())
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

#Sets the initial Page
page_manager = PageManager(MainMenuPage())

running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Pass the event to the current page
            page_manager.handle_event(event)
        elif event.type == pygame.KEYDOWN:
            # Pass the event to the current page
            page_manager.handle_event(event)
    # Render the current page
    page_manager.render(screen)

    pygame.display.flip()

print("Thanks for using Franks Chemical Simulator")
#you could encapsulate the state for each menu in a class or something
#and the game loop can pick what class to render based on the state

#pickle could be sav cuz u can make a class or something that holds 
#the entire state and then u can serialize the python class with pickle

#u should add some loggin' when u do calls that try and load files

"""
something like that yeah
well youd still need the elifs probably to choose what page is being render
but itd be like one line elifs
your you could use a dict or something, like a dict mapping state => page
thatd be more tasty than elifs probs
"""
