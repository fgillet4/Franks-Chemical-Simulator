import pygame
import csv
import pickle
import math
from enum import Enum
import os
flowsheet_version = "Flowsheet Simulator v.1.0.0"
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128, 128)
BLUE = (0, 0, 255)

# Initialize Pygame
pygame.init()

# Set up the Pygame window
WINDOW_SIZE = (1800, 900)
screen_dimensions = WINDOW_SIZE
screen_length = WINDOW_SIZE[0]
screen_height = WINDOW_SIZE[1]
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
                        self.manager.running = False
                        self.in_quit= True
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
            self.manager.running = False
        
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
            self.manager.go_to(NewFlowsheetPage(page_manager=self.manager))
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
# Define the Run Flow sheet Page
class RunFlowsheetSimulation(Page):
    def __init__(self, filename, screen, page_manager):
        super().__init__()
        print("Initializing RunFlowsheetSimulation...")
        self.page_manager = page_manager
        self.filename = filename
        self.flowsheet = Flowsheet("My Flowsheet")
        self.grid = None
        self.paused = False
        self.save_as = False
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        self.clock = pygame.time.Clock()
        self.paused = False
        self.flowsheet_renderer = FlowsheetRenderer(self.flowsheet)
        self.placing_pump = False
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
                        else:
                            self.placing_pump = False
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
                    print("Game Saved")
                    with open(self.filename, 'wb') as f:
                        pickle.dump(self.flowsheet, f)
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
                    self.page_manager.change_page(MainMenuPage())

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
                                dragged_block = block
                                break
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.left_click = False
                    dragged_block = None
        else:
            if self.left_click and dragged_block is not None:
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
        elif block_type == BlockType.Valve:
            block = Valve(name=f"{block_type.name}_{len(self.flowsheet.blocks)}", position=(x, y))
        elif block_type == BlockType.FlashTank:
            block = FlashTank(name=f"{block_type.name}_{len(self.flowsheet.blocks)}", position=(x, y))
        elif block_type == BlockType.DistillationColumn:
            block = DistillationColumn(name=f"{block_type.name}_{len(self.flowsheet.blocks)}", position=(x, y))

        self.flowsheet.blocks.append(block)
        print(f"Added {block.name} at position {block.position}")  # Add this line for debugging

        block.size = (150, 50)

        block_info = {"type": block_type.value, "rect": pygame.Rect(x, y, 150, 50), "instance": block}
        self.block_list.append(block_info)
    def draw_blocks(self):
        for block in self.block_list:
            if block["type"] == BlockType.CentrifugalPump.value:
                self.renderer.draw_centrifugal_pump(self.screen, block["rect"].center)
            else:
                pygame.draw.rect(self.screen, self.BLUE, block["rect"])
                text = self.font.render(block["type"], True, self.BLACK)
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
        self.menu_rects = [
        pygame.Rect(button_start_x/2-button_width/2, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x/2-button_width/2, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x/2-button_width/2, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x/2-button_width/2, button_start_y + (button_height + button_padding) * 3, button_width, button_height),
        pygame.Rect(button_start_x/2-button_width/2, button_start_y + (button_height + button_padding) * 4, button_width, button_height),
        pygame.Rect(button_start_x/2-button_width/2, button_start_y + (button_height + button_padding) * 5, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y, button_width, button_height)
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
        text = font.render("Equipment Sizing", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        for i, rect in enumerate(self.menu_rects):
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
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
        text = font.render("Types of Process Vessels", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        for i, rect in enumerate(self.menu_rects):
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
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
        text = font.render("Types of Pressure Vessels", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        for i, rect in enumerate(self.menu_rects):
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            pygame_screen.blit(text, text_rect)
# Define the Open Tanks Sizing Page 1
class OpenTankSizingPage1(Page):
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
        tank_x = screen_length // 2 - tank_width//2
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
        tank_x = screen_length // 2 - tank_width//2
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
    pass
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
        text = font.render("Estimate Flowsheet Safety", True, BLACK)
        text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
        screen.blit(text, text_rect)
        for i, rect in enumerate(self.menu_rects):
            pygame.draw.rect(pygame_screen, BLACK, rect, 2)
            text = font.render(self.menu_texts[i], True, BLACK)
            text_rect = text.get_rect(center=rect.center)
            pygame_screen.blit(text, text_rect)
# Define Flowsheet Safety Estimation Page
class EstimateFlowsheetSafetyPage(Page):
    pass
# Define Chemical Safety Page
class FindChemicalSafetyPage(Page):
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
class RuptureDiskPage(Page):
    pass
# Define Releif Valves Page
class ReleifValvesPage(Page):
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
            screen_dimensions = (WINDOW_SIZE[0], WINDOW_SIZE[1])
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
    def __init__(self) -> None:
        super().__init__()
        self.menu_rects = []
        self.menu_texts = []
        self.in_back = False
        self.manager = page_manager
    def handle_event(self, event):
        pass
    def render(self, pygame_screen):
        pass

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
    def draw_online_instrumentation(screen, center_pos, label):
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
    def draw_distillation_column(screen, center_pos, label, column_width, column_height, tray_thickness=2):
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
    def draw_shell_and_tube_heat_exchanger(screen, center_pos):
        # Draw shell
        pygame.draw.ellipse(screen, BLACK, (center_pos[0] - 50, center_pos[1] - 20, 100, 40), 2)

        # Draw tube bundle
        for i in range(5):
            pygame.draw.line(screen, BLACK, (center_pos[0] - 40 + i * 20, center_pos[1] - 15), (center_pos[0] - 40 + i * 20, center_pos[1] + 15), 2)

        # Draw channel heads
        pygame.draw.rect(screen, BLACK, (center_pos[0] - 60, center_pos[1] - 20, 10, 40), 2)
        pygame.draw.rect(screen, BLACK, (center_pos[0] + 50, center_pos[1] - 20, 10, 40), 2)
    # Draw a heat exchanger
    def draw_heat_exchanger(screen, center_pos, radius=20):
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
    def draw_filter_press(screen, center_pos, label, column_width, column_height, tray_thickness=2):
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
    def draw_dome_roof_tank(screen, center_pos, width, height, color=BLACK):
        # Draw the tank body (rectangle)
        tank_rect = pygame.Rect(center_pos[0] - width / 2, center_pos[1], width, height)
        pygame.draw.rect(screen, color, tank_rect, 2)

        # Draw the dome (half-circle)
        dome_rect = pygame.Rect(center_pos[0] - width / 2, center_pos[1] - height / 2, width, height)
        pygame.draw.arc(screen, color, dome_rect, 0, math.pi, 2)
    # Draw a centrifugal pump
    def draw_centrifugal_pump(screen, center_pos, radius=20):
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
    def draw_storage_tank(screen, center_pos):
        pygame.draw.ellipse(screen, BLACK, (center_pos[0] - 30, center_pos[1] - 60, 60, 30), 2)
        pygame.draw.rect(screen, BLACK, (center_pos[0] - 30, center_pos[1] - 30, 60, 60), 2)
        pygame.draw.ellipse(screen, BLACK, (center_pos[0] - 30, center_pos[1], 60, 30), 2)

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
class Tank(Block):
    def __init__(self, name, position):
        super().__init__(name)
        self.position = position
        self.x, self.y = position

class CentrifugalPump(Block):
    def __init__(self, name, position):
        super().__init__(name)
        self.position = position
        self.x, self.y = position

class Valve(Block):
    pass

class FlashTank(Block):
    pass

class DistillationColumn(Block):
    pass

class StrippingColumn(Block):
    pass

class AbsorptionColumn(Block):
    pass

# Enums for block types
class BlockType(Enum):
    Tank = "Tank"
    CentrifugalPump = "Centrifugal Pump"
    Valve = "Valve"
    FlashTank = "Flash Tank"
    DistillationColumn = "Distillation Column"  # Add this line
    StrippingColumn = "Stripping Column"
    AbsorptionColumn = "Absorption Column"
    StorageTank = "Storage Tank"

# Create a ui_manager object
#ui_manager = pygame_gui.UIManager(screen_dimensions)

# Create the MainMenuPage object
#main_menu_page = MainMenuPage(screen_dimensions=screen_dimensions, ui_manager=ui_manager)

# Create the MainMenuPage object
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
        page_manager.change_page(RunFlowsheetSimulation(filename, screen, page_manager).run())

    # Render the current page
    page_manager.render(screen)
    pygame.display.flip()
    clock.tick(60)


print("Thanks for using Franks Chemical Simulator")
#you could encapsulate the state for each menu in a class or something
#and the game loop can pick what class to render based on the state

#pickle could be sav cuz u can make a class or something that holds 
#the entire state and then u can serialize the python class with pickle

#u should add some loggin' when u do calls that try and load files
