def main_menu():
    import pygame
    import os
    import csv
    #Get current working directory
    current_dir = os.getcwd()
    # Define some colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (128, 128, 128, 128)

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

    # Define Menu Buttons
    menu_rects = [
        pygame.Rect(button_start_x-15, button_start_y, button_width+30, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x-10, button_start_y + (button_height + button_padding) * 2, button_width+20, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 4, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 5, button_width, button_height)
    ]
    menu_texts = [
        "Flowsheet Simulation",
        "Equipment Sizing",
        "Process Economics",
        "Process Safety",
        "Physical Properties",
        "Quit"
    ]
    # Define Simulation Buttons
    simulation_submenu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height)
    ]
    simulation_submenu_texts = [
        "New Flowsheet",
        "Load Flowsheet",
        "Back"
    ]
    # Define Equiptment Buttons
    equiptment_sizing_submenu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 4, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 5, button_width, button_height)
    ]
    equiptment_sizing_submenu_texts = [
        "Heat Exchangers",
        "Separations",
        "Pumps",
        "Reactors",
        "Controls",
        "Back"
    ]
    # Define Heat exchanger buttons
    heat_exchangers_submenu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 4, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 5, button_width, button_height)
    ]
    heat_exchangers_submenu_texts = [
        "Shell and Tube",
        "Plate",
        "Spiral",
        "Finned Tube",
        "Double Pipe",
        "Back"
    ]
    # Define Seps buttons
    separations_submenu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height)
    ]
    separations_submenu_texts = [
        "Gas",
        "Liquid",
        "Solid",
        "Back"
    ]
    # Define pump buttons
    pumps_submenu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height),
    ]
    pumps_submenu_texts = [
        "Centrifugal",
        "Positive Displacement",
        "Ejectors",
        "Back"
    ]
    # Define Centrifugal pump buttons
    centrifugal_submenu_rects =[
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 4, button_width, button_height)
    ]
    centrifugal_submenu_texts = [
        "Multi-Stage",
        "Single Stage",
        "Magnetic Drive",
        "Submersible",
        "Back"
    ]
    # Define Pos Displacement pump buttons
    pos_displacement_submenu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 4, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 5, button_width, button_height)
    ]
    pos_displacement_submenu_texts = [
        "Piston",
        "Rotary lobe",
        "Diaphragm",
        "Peristaltic",
        "Screw",
        "Back"


    ]
    # Define Reactor Buttons
    reactors_submenu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height)
    ]
    reactors_submenu_texts = [
        "CSTR",
        "PFR",
        "PBR",
        "Back"
    ]
    # Define Controls Buttons
    controls_submenu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 4, button_width, button_height)
    ]
    controls_submenu_texts = [
        "Valves",
        "Sensors & Instr.",
        "Transfer Fn",
        "Fluid Handling",
        "Back"
    ]
    # Define Fluid Handling
    fluid_handling_submenu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 4, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 5, button_width, button_height)
    
    ]
    fluid_handling_submenu_texts = [
        "Pipes",
        "Bends",
        "Orifices",
        "Tees",
        "Vena Contracta",
        "Back"

    ]
    # Define Sensor Buttons
    sensors_submenu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 4, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 5, button_width, button_height)
    ]
    sensors_submenu_texts = [
        "Temperature",
        "Pressure",
        "Flow",
        "Level",
        "Composition",
        "Back"
    ]
    # Define Chemical Composition Analyzer Buttons
    composition_sensor_submenu_recs = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 4, button_width, button_height)

    ]
    composition_sensor_submenu_texts = [
        "pH",
        "Conductivity",
        "FT-IR",
        "Gas Alarms",
        "Back"
    ]
    # Define Capital Cost Estimation Buttons
    capital_cost_estimation_submenu_rects = [
        pygame.Rect(button_start_x-100, button_start_y, button_width+200, button_height+20),
        pygame.Rect(button_start_x-125, button_start_y + button_height + button_padding, button_width+250, button_height+20),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height)
    ]
    capital_cost_estimation_submenu_texts = [
        "Estimate Flowsheet Capital Cost",
        "Edit Capital Cost Estimation Factors",
        "Back"
    ]
    # Define Process Safety Buttons
    process_safety_submenu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 3, button_width, button_height)
    ]
    process_safety_submenu_texts = [
        "Estimate Flowsheet Safety",
        "Find Chemical Safety Properties",
        "Safety Instr.",
        "Back"
    ]
    #Define Safety Instrumentation Buttons
    safety_instrumentation_submenu_recs = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height)
    ]
    safety_instrumentation_submenu_texts = [
        "Rupture Disks",
        "Releif Valves",
        "Back"
    ]
    #Define Physicals Properties
    physical_properties_submenu_rects = [
        pygame.Rect(button_start_x, button_start_y, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + button_height + button_padding, button_width, button_height),
        pygame.Rect(button_start_x, button_start_y + (button_height + button_padding) * 2, button_width, button_height)
    ]
    physical_properties_submenu_texts = [
        "Add a Chemical",
        "Find a Chemical",
        "Back"
    ]
    # Set up Menu Layers with True False Switching
    
    #Main Menu Layer
    in_main_menu = True

    #Flowsheet Simulation Layer
    in_flowsheet_sim = False
    in_new_flowsheet = False
    in_load_flowsheet = False

    # Capital Cost Estimation Layer
    in_capital_cost = False

    #Physical Properties Layer
    in_physical_properties = False

    #Equitment Sizing Sublayer
    in_equiptment_sizing = False

    #Separations Sizing Sublayer
    in_separations = False
    in_gas_seps = False
    in_liquid_seps = False
    in_solid_seps = False

    #Pump sublayer 
    in_pumps = False

    #Centrifugal Pump sublayer
    in_centrifugal_pumps = False
    in_multistage_cent_pump = False
    in_single_drive_cent_pump = False
    in_magnetic_drive_pump = False

    #Postive Displacement Pumps Sublayer
    in_pos_displacement_pump = False
    in_rotary_lobe_pump = False
    in_piston_pump = False
    in_screw_pump = False

    # Jet style steam ejector pumps
    in_ejectors = False

    #Sublayer for Heat exchangers
    in_heat_exchangers = False

    #Sublayer for Reactors
    in_reactors = False
    in_cstr = False
    in_pfr = False
    in_pbr = False

    # Sublayer for Controls
    in_controls = False
    in_valves = False
    in_sensors = False
    in_composition_sensors = False
    in_temperature_sensors = False
    in_pressure_sensors = False
    in_level_sensors = False
    in_flow_sensors = False
    in_ph_sensors = False
    in_ft_ir_sensors = False
    in_gas_sensors = False
    in_conductivity_sensors = False
    in_transfer_functions = False
    in_hand_valves = False
    in_control_valves = False
    in_fluid_handling = False

    # Process Safety Sublayer
    in_process_safety = False
    in_estimate_flowsheet_safety = False
    in_chemical_safety = False
    in_safety_instrumentation = False
    in_rupture_disk = False
    in_releif_valves = False

    # Main loop
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if in_main_menu:
                    # Check if a menu button was clicked
                    for i, rect in enumerate(menu_rects):
                        if rect.collidepoint(event.pos):
                            if i == 0:
                                # "Flowsheet Simulation" button clicked
                                in_flowsheet_sim = True
                                in_main_menu = False
                            elif i == 1:
                                # "Equipment Sizing" button clicked
                                print("Equipment Sizing")
                                in_equiptment_sizing = True
                                in_main_menu = False
                            elif i == 2:
                                # "Capital Cost Estimation" button clicked
                                print("Capital Cost Estimation")
                                in_capital_cost = True
                                in_main_menu = False
                            elif i == 3:
                                # "Chemical Safety" button clicked
                                print("Chemical Safety")
                                in_process_safety = True
                                in_main_menu = False
                            elif i == 4:
                                # "Physical Properties" button clicked
                                print("Physical Properties")
                                in_physical_properties = True
                                in_main_menu = False
                            elif i == 5:
                                #Quit button clicked
                                print("Quit")
                                pygame.quit()
                # Handle Flowsheet  Clicked Buttons
                elif in_flowsheet_sim:
                    # Check if simulation sub-menu button was clicked
                    for i, rect in enumerate(simulation_submenu_rects):
                        if rect.collidepoint(event.pos):
                            if i == 0:
                                # "New Flowsheet" button clicked
                                print("New Flowsheet")
                                in_new_flowsheet = True
                                # Check if the directory exists
                                if not os.path.isdir(os.path.join(current_dir, 'Flowsheets')):
                                # Create the directory
                                    os.makedirs(os.path.join(current_dir, 'Flowsheets'))
                                if in_new_flowsheet:
                                    overlay = pygame.Surface((1600, 800))
                                    overlay.fill(GRAY)
                                    overlay.set_alpha(128)
                                    # Create an input box for the filename
                                    input_rect = pygame.Rect(100, 300, 400, 50)
                                    input_text = ""
                                    # Draw the input box and text
                                    screen.blit(overlay, (0, 0))
                                    pygame.draw.rect(screen, WHITE, input_rect)
                                    pygame.display.update(input_rect)
                                    while in_new_flowsheet:
                                        for event in pygame.event.get():
                                            if event.type == pygame.KEYDOWN:
                                                #if event.key == pygame.K_ESCAPE:
                                                #    go_back = True
                                                #    in_new_flowsheet = False
                                                if event.key == pygame.K_RETURN:
                                                    # Save the file and return to the game
                                                    filename = input_text.strip() + ".csv"
                                                    with open(filename, "w", newline="") as f:
                                                        writer = csv.writer(f)
                                                        #for key, value in game_state.items():
                                                        #    writer.writerow([key, value])
                                                    print("Game saved as", filename)
                                                    in_new_flowsheet = False
                                                elif event.key == pygame.K_BACKSPACE:
                                                    input_text = input_text[:-1]
                                                    input_surface = font.render(input_text, True, BLACK)
                                                    pygame.draw.rect(screen, WHITE, input_rect)
                                                    screen.blit(input_surface, (input_rect.x + 5, input_rect.y + 5))
                                                    pygame.draw.rect(screen, WHITE, input_rect, 2)
                                                    pygame.display.update(input_rect)
                                                elif event.key not in (pygame.K_LSHIFT, pygame.K_RSHIFT, pygame.K_ESCAPE):
                                                    input_text += event.unicode
                                                    input_surface = font.render(input_text, True, BLACK)
                                                    pygame.draw.rect(screen, WHITE, input_rect)
                                                    screen.blit(input_surface, (input_rect.x + 5, input_rect.y + 5))
                                                    pygame.draw.rect(screen, WHITE, input_rect, 2)
                                                    pygame.display.update(input_rect)
                                from run_simulation import run_simulation
                                run_simulation()
                            elif i == 1:
                                # "Load Flowsheet" button clicked
                                print("Load Flowsheet")
                            elif i == 2:
                                # Back button is clicked
                                print("Back")
                                in_main_menu = True
                                in_flowsheet_sim = False
                # Handle Heat Exchanger Clicked Buttons
                elif in_heat_exchangers:
                    for i, rect in enumerate(heat_exchangers_submenu_rects):
                        if rect.collidepoint(event.pos):
                            if i == 0:
                                print("Shell and Tube Heat Exchangers")
                            elif i == 1:
                                print("Plate Heat Exchangers")
                            elif i == 2:
                                print("Spiral Heat Exchangers")
                            elif i == 3:
                                print("Finned Tube Heat Exchangers")
                            elif i == 4:
                                print("Double Pipe Heat Exchanger")
                            elif i == 5:
                                in_equiptment_sizing = True
                                in_heat_exchangers = False
                # Handle Separations Clicked Buttons
                elif in_separations:
                    for i, rect in enumerate(separations_submenu_rects):
                        if rect.collidepoint(event.pos):
                            if i == 0:
                                print("Gas")
                            elif i == 1:
                                print("Liquid")
                            elif i == 2:
                                print("Solid")
                            elif i == 3:
                                print("Back")
                                in_equiptment_sizing = True
                                in_separations = False
                # Handle Pumps Clicked Buttons
                elif in_pumps:
                    for i, rect in enumerate(pumps_submenu_rects):
                        if rect.collidepoint(event.pos):
                            if i == 0:
                                print("Centrifugal")
                                in_pumps = False
                                in_centrifugal_pumps = True
                            elif i == 1:
                                print("Positive Displacement")
                                in_pumps = False
                                in_pos_displacement_pump = True
                            elif i == 2:
                                print("Ejectors")
                            elif i == 3:
                                print("Back")
                                in_equiptment_sizing = True
                                in_pumps = False
                # Handle Fluid Handling Clicked Buttons
                elif in_fluid_handling:
                    for i, rect in enumerate(fluid_handling_submenu_rects):
                        if rect.collidepoint(event.pos):
                            if i == 0:
                                print("Pipes")
                            elif i == 1:
                                print("Bends")
                            elif i == 2:
                                print("Orifices")
                            elif i == 3:
                                print("Tees")
                            elif i == 4:
                                print("Vena Contracta")
                            elif i == 5:
                                print("Back")
                                in_fluid_handling = False
                                in_controls = True
                # Handle Centrifugal Pump Clicked Buttons
                elif in_centrifugal_pumps:
                    for i, rect in enumerate(centrifugal_submenu_rects):
                        if rect.collidepoint(event.pos):
                            if i == 0:
                                print("Multi-Stage")
                            elif i == 1:
                                print("Single Stage")
                            elif i == 2:
                                print("Magnetic Drive")
                            elif i == 3:
                                print("Submersible")
                            elif i == 4:
                                print("Back")
                                in_pumps = True
                                in_centrifugal_pumps = False
                # Handle Positive Displacement Pump Clicked Buttons
                elif in_pos_displacement_pump:
                    for i, rect in enumerate(pos_displacement_submenu_rects):
                        if rect.collidepoint(event.pos):
                            if i == 0:
                                print("Piston")
                            elif i == 1:
                                print("Rotary lobe")
                            elif i == 2:
                                print("Diaphragm")
                            elif i == 3:
                                print("Peristaltic")
                            elif i == 4:
                                print("Screw")
                            elif i == 5:
                                print("Back")
                                in_pos_displacement_pump = False
                                in_pumps = True
                # Handle Reactor Clicked Buttons
                elif in_reactors:
                    for i, rect in enumerate(reactors_submenu_rects):
                        if rect.collidepoint(event.pos):
                            if i == 0:
                                print("CSTR")
                            elif i == 1:
                                print("PFR")
                            elif i == 2:
                                print("PBR")
                            elif i == 3:
                                print("Back")
                                in_equiptment_sizing = True
                                in_reactors = False
                # Handle Controls Clicked Buttons
                elif in_controls:
                    for i, rect in enumerate(controls_submenu_rects):
                        if rect.collidepoint(event.pos):
                            if i == 0:
                                print("Valves")
                            elif i == 1:
                                print("Sensors & Instr.")
                                in_sensors = True
                                in_controls = False
                            elif i == 2:
                                print("Transfer Fn")
                            elif i == 3:
                                print("Fluid Handling")
                                in_fluid_handling = True
                                in_controls = False
                            elif i == 4:
                                print("Back")
                                in_equiptment_sizing = True
                                in_controls = False
                # Handle Sensor Clicked Buttons
                elif in_sensors:
                    # Check if Equiptment Sizing sub-menu button was clicked
                    for i, rect in enumerate(sensors_submenu_rects):
                        if rect.collidepoint(event.pos):
                            if i == 0:
                                print("Temperature")
                            elif i == 1:
                                print("Pressure")
                            elif i == 2:
                                print("Flow")
                            elif i == 3:
                                print("Level")
                            elif i == 4:
                                print("Composition")
                                in_sensors = False
                                in_composition_sensors = True
                            elif i == 5:
                                print("Back")
                                in_sensors = False
                                in_controls = True
                # Handle Compositions Analyzer Clicked Buttons
                elif in_composition_sensors:
                    # Check if Equiptment Sizing sub-menu button was clicked
                    for i, rect in enumerate(composition_sensor_submenu_recs):
                        if rect.collidepoint(event.pos):
                            if i == 0:
                                print("pH")
                            elif i == 1:
                                print("Conductivity")
                            elif i == 2:
                                print("FT-IR")
                            elif i == 3:
                                print("Gas Alarms")
                            elif i == 4:
                                print("Back")
                                in_composition_sensors = False
                                in_sensors = True
                # Handle Equiptment Sizing Clicked Buttons
                elif in_equiptment_sizing:
                    # Check if Equiptment Sizing sub-menu button was clicked
                    for i, rect in enumerate(equiptment_sizing_submenu_rects):
                        if rect.collidepoint(event.pos):
                            if i == 0:
                                # "Heat Exchanger" button clicked
                                print("Heat Exchanger")
                                in_heat_exchangers = True
                                in_equiptment_sizing = False
                            elif i == 1:
                                # "Separations Column button clicked
                                print("Separations")
                                in_separations = True
                                in_equiptment_sizing = False
                            elif i == 2:
                                # Pump Column
                                print("Pumps")
                                in_pumps = True
                                in_equiptment_sizing = False
                            elif i == 3:
                                print("Reactors")
                                in_reactors = True
                                in_equiptment_sizing = False
                            elif i == 4:
                                print("Controls")
                                in_controls = True
                                in_equiptment_sizing = False
                            elif i == 5:
                                # Back button is clicked
                                print("Back")
                                in_main_menu = True
                                in_equiptment_sizing = False
                # Handle Capital Cost Clicked Buttons
                elif in_capital_cost:
                    # Check if Capital Cost Estimation sub-menu button was clicked
                    for i, rect in enumerate(capital_cost_estimation_submenu_rects):
                        if rect.collidepoint(event.pos):
                            if i == 0:
                                # "New Flowsheet" button clicked
                                print("Estimate Flowsheet Capital Cost")
                            elif i == 1:
                                # "Load Flowsheet" button clicked
                                print("Edit Capital Cost Estimation Factors")
                            elif i == 2:
                                # Back button is clicked
                                print("Back")
                                in_main_menu = True
                                in_capital_cost = False
                # Handle Process Safety Clicked Buttons
                elif in_process_safety:
                    #Check if Chemical Safety sub-menu was clicked
                    for i, rect in enumerate(process_safety_submenu_rects):
                        if rect.collidepoint(event.pos):
                            if i == 0:
                                # "New Flowsheet" button clicked
                                print("Estimate Flowsheet Safety")
                            elif i == 1:
                                # "Load Flowsheet" button clicked
                                print("Find Chemical Safety Properties")
                            elif i == 2:
                                #Safety Instrumentation
                                print("Safety Instrumentation")
                                in_safety_instrumentation = True
                                in_process_safety = False
                            elif i == 3:
                                # Back button is clicked
                                print("Back")
                                in_main_menu = True
                                in_process_safety = False
                # Handle Safety Instrumentation Clicked Buttons
                elif in_safety_instrumentation:
                    #Check if Chemical Safety sub-menu was clicked
                    for i, rect in enumerate(safety_instrumentation_submenu_recs):
                        if rect.collidepoint(event.pos):
                            if i == 0:
                                # "New Flowsheet" button clicked
                                print("Rupture Disks")
                            elif i == 1:
                                # "Load Flowsheet" button clicked
                                print("Releif Valves")
                            elif i == 2:
                                # Back button is clicked
                                print("Back")
                                in_process_safety = True
                                in_safety_instrumentation = False
                # Handle Physical Properties Clicked Buttons
                elif in_physical_properties:
                    #Check if Chemical Safety sub-menu was clicked
                    for i, rect in enumerate(physical_properties_submenu_rects):
                        if rect.collidepoint(event.pos):
                            if i == 0:
                                # "New Flowsheet" button clicked
                                print("Add a Chemical")
                            elif i == 1:
                                # "Load Flowsheet" button clicked
                                print("Find a Chemical")
                            elif i == 2:
                                # Back button is clicked
                                print("Back")
                                in_main_menu = True
                                in_physical_properties = False            
                # 
        # Draw the UI
        screen.fill(WHITE)
        # Draw Main Menu
        if in_main_menu:
            # Draw the main menu
            text = font.render("Welcome to Frank's Chemical Process Simulator", True, BLACK)
            text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
            screen.blit(text, text_rect)

            for i, rect in enumerate(menu_rects):
                pygame.draw.rect(screen, BLACK, rect, 2)
                text = font.render(menu_texts[i], True, BLACK)
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)
        #Draw Menu for Flowsheet Simulator
        elif in_flowsheet_sim:
            # Draw the flowsheet simulation sub-menu
            text = font.render("Flowsheet Simulation", True, BLACK)
            text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
            screen.blit(text, text_rect)

            for i, rect in enumerate(simulation_submenu_rects):
                pygame.draw.rect(screen, BLACK, rect, 2)
                text = font.render(simulation_submenu_texts[i], True, BLACK)
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)
        #Draw Menu for Equiptment Sizing
        elif in_equiptment_sizing:
            # Draw the equipment sizing sub-menu
            text = font.render("Equipment Sizing", True, BLACK)
            text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
            screen.blit(text, text_rect)

            for i, rect in enumerate(equiptment_sizing_submenu_rects):
                pygame.draw.rect(screen, BLACK, rect, 2)
                text = font.render(equiptment_sizing_submenu_texts[i], True, BLACK)
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)
        #Draw Menu For Heat Exchangers
        elif in_heat_exchangers:
            # Draw the equipment sizing sub-menu
            text = font.render("Heat Exchangers", True, BLACK)
            text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
            screen.blit(text, text_rect)

            for i, rect in enumerate(heat_exchangers_submenu_rects):
                pygame.draw.rect(screen, BLACK, rect, 2)
                text = font.render(heat_exchangers_submenu_texts[i], True, BLACK)
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)
        # Draw Menu For Separations
        elif in_separations:
            # Draw the equipment sizing sub-menu
            text = font.render("Choose Separation Phase", True, BLACK)
            text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
            screen.blit(text, text_rect)
            for i, rect in enumerate(separations_submenu_rects):
                pygame.draw.rect(screen, BLACK, rect, 2)
                text = font.render(separations_submenu_texts[i], True, BLACK)
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)
        # Draw Menu for Pumps
        elif in_pumps:
            text = font.render("Select Pump Type", True, BLACK)
            text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
            screen.blit(text, text_rect)
            for i, rect in enumerate(pumps_submenu_rects):
                pygame.draw.rect(screen, BLACK, rect, 2)
                text = font.render(pumps_submenu_texts[i], True, BLACK)
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)
        # Draw Menu for Centrifugal Pumps
        elif in_centrifugal_pumps:
            text = font.render("Select Centrifugal Pump Type", True, BLACK)
            text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
            screen.blit(text, text_rect)
            for i, rect in enumerate(centrifugal_submenu_rects):
                pygame.draw.rect(screen, BLACK, rect, 2)
                text = font.render(centrifugal_submenu_texts[i], True, BLACK)
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)
        # Draw Menu for Positive Displacement Pumps
        elif in_pos_displacement_pump:
            text = font.render("Select Positive Displacement Pump Type", True, BLACK)
            text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
            screen.blit(text, text_rect)
            for i, rect in enumerate(pos_displacement_submenu_rects):
                pygame.draw.rect(screen, BLACK, rect, 2)
                text = font.render(pos_displacement_submenu_texts[i], True, BLACK)
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)
        # Draw Menu for Reactors
        elif in_reactors:
            text = font.render("Select Reactor Type", True, BLACK)
            text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
            screen.blit(text, text_rect)
            for i, rect in enumerate(reactors_submenu_rects):
                pygame.draw.rect(screen, BLACK, rect, 2)
                text = font.render(reactors_submenu_texts[i], True, BLACK)
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)
        # Draw Menu For Controls
        elif in_controls:
            text = font.render("Select Controls Type", True, BLACK)
            text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
            screen.blit(text, text_rect)
            for i, rect in enumerate(controls_submenu_rects):
                pygame.draw.rect(screen, BLACK, rect, 2)
                text = font.render(controls_submenu_texts[i], True, BLACK)
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)
        # Draw Menu for Sensors
        elif in_sensors:
            # Draw the capital cost estimation sub-menu
            text = font.render("Select Sensor Type", True, BLACK)
            text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
            screen.blit(text, text_rect)

            for i, rect in enumerate(sensors_submenu_rects):
                pygame.draw.rect(screen, BLACK, rect, 2)
                text = font.render(sensors_submenu_texts[i], True, BLACK)
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)
        #Draw Menu for Capital Cost Estimation
        elif in_capital_cost:
            # Draw the capital cost estimation sub-menu
            text = font.render("Capital Cost Estimation", True, BLACK)
            text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
            screen.blit(text, text_rect)

            for i, rect in enumerate(capital_cost_estimation_submenu_rects):
                pygame.draw.rect(screen, BLACK, rect, 2)
                text = font.render(capital_cost_estimation_submenu_texts[i], True, BLACK)
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)
        #Draw Menu for Chemical Safety
        elif in_process_safety:
            # Draw the chemical safety sub-menu
            text = font.render("Chemical Process Safety", True, BLACK)
            text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
            screen.blit(text, text_rect)

            for i, rect in enumerate(process_safety_submenu_rects):
                pygame.draw.rect(screen, BLACK, rect, 2)
                text = font.render(process_safety_submenu_texts[i], True, BLACK)
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)     
        #Draw menu for Physical Properties
        elif in_physical_properties:
            # Draw the physical properties sub-menu
            text = font.render("Physical Properties", True, BLACK)
            text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
            screen.blit(text, text_rect)

            for i, rect in enumerate(physical_properties_submenu_rects):
                pygame.draw.rect(screen, BLACK, rect, 2)
                text = font.render(physical_properties_submenu_texts[i], True, BLACK)
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)
        # Draw Menu for Fluid Handling
        elif in_fluid_handling:
            # Draw the capital cost estimation sub-menu
            text = font.render("Select Fluid Handling Equiptiment", True, BLACK)
            text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
            screen.blit(text, text_rect)

            for i, rect in enumerate(fluid_handling_submenu_rects):
                pygame.draw.rect(screen, BLACK, rect, 2)
                text = font.render(fluid_handling_submenu_texts[i], True, BLACK)
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)
        # Draw Menu for Composition Analyzers
        elif in_composition_sensors:
            # Draw the capital cost estimation sub-menu
            text = font.render("Select Chemical Composition Analyzer Type", True, BLACK)
            text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
            screen.blit(text, text_rect)

            for i, rect in enumerate(composition_sensor_submenu_recs):
                pygame.draw.rect(screen, BLACK, rect, 2)
                text = font.render(composition_sensor_submenu_texts[i], True, BLACK)
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)
        # Draw Menu for Safety Instrumentation
        elif in_safety_instrumentation:
            # Draw the capital cost estimation sub-menu
            text = font.render("Select Safetey Instrumentation Type", True, BLACK)
            text_rect = text.get_rect(center=(WINDOW_SIZE[0] / 2, button_start_y / 2))
            screen.blit(text, text_rect)

            for i, rect in enumerate(safety_instrumentation_submenu_recs):
                pygame.draw.rect(screen, BLACK, rect, 2)
                text = font.render(safety_instrumentation_submenu_texts[i], True, BLACK)
                text_rect = text.get_rect(center=rect.center)
                screen.blit(text, text_rect)

        # Update Frame
        pygame.display.flip()

main_menu()
