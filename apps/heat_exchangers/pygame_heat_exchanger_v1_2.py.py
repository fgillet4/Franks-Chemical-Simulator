import pygame

# Constants and calculation function

h = 100

# Pygame initialization
pygame.init()
pygame.display.set_caption("Heat Transfer Coefficient Calculator")

# Set up the Pygame window
width, height = 800, 600
screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
bg_color = (255, 255, 255)
font = pygame.font.SysFont(None, 30)

# Create input fields and create the "Calculate" button
# Create input fields
labels = ["Thermal conductivity of tube wall (W/mK): ", "Diameter of tube (mm): ", "Length of tube (mm): ",
          "Density of vapor (kg/m^3): ", "Viscosity of vapor (Pa.s): ", "Density of liquid (kg/m^3): ",
          "Viscosity of liquid (Pa.s): ", "Surface tension (N/m): ", "Acceleration due to gravity (m/s^2): "]
inputs = []
for i in range(len(labels)):
    input_box = pygame.Rect(350, 50 + i * 50, 200, 32)
    inputs.append({"rect": input_box, "color": pygame.Color("white"), "text": "", "label": labels[i]})

# Add images and icons
icon = pygame.image.load("icon.png")
background = pygame.image.load("background.jpg")

# Add a "Calculate" button
button = pygame.Rect(350, 500, 100, 50)
button_color = pygame.Color("blue")
button_text = font.render("Calculate", True, (255, 255, 255))


# Add a "Quit" button
quit_button = pygame.Rect(650, 500, 100, 50)
quit_button_color = pygame.Color("red")
quit_button_text = font.render("Quit", True, (255, 255, 255))

# Main loop
# Main loop
while True:
    # Get user inputs, calculate the heat transfer coefficient, and draw the UI
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            for inp in inputs:
                if inp["rect"].collidepoint(event.pos):
                    inp["color"] = pygame.Color("yellow")
                else:
                    inp["color"] = pygame.Color("white")
            if button.collidepoint(event.pos):
                # Calculate the heat transfer coefficient
                k_tube = float(inputs[0]["text"])
                d_tube = float(inputs[1]["text"])
                l_tube = float(inputs[2]["text"])
                rho_v = float(inputs[3]["text"])
                mu_v = float(inputs[4]["text"])
                rho_l = float(inputs[5]["text"])
                mu_l = float(inputs[6]["text"])
                sigma = float(inputs[7]["text"])
                g = float(inputs[8]["text"])
                calculate_h()
            elif quit_button.collidepoint(event.pos):
                pygame.quit()
                quit()

        if event.type == pygame.KEYDOWN:
            for inp in inputs:
                if inp["color"] == pygame.Color("yellow"):
                    if event.key == pygame.K_BACKSPACE:
                        inp["text"] = inp["text"][:-1]
                    else:
                        inp["text"] += event.unicode

        if event.type == pygame.VIDEORESIZE:
            # Resize the window and the UI elements
            width, height = event.size
            screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
            bg_color = (255, 255, 255)
            font_size = min(int(height / 20), 40)
            font = pygame.font.SysFont(None, font_size)
            button = pygame.Rect(50, int(height / 2.5), 150, int(height / 15))
            quit_button = pygame.Rect(int(width / 1.3), int(height / 1.2), int(width / 8), int(height / 15))
            for i, inp in enumerate(inputs):
                inp["rect"] = pygame.Rect(50, int(height / 10) + i * int(height / 15), width - 100, int(height / 20))

    # Draw the UI
    screen.fill(bg_color)
    for inp in inputs:
        label = font.render(inp["label"], True, (0, 0, 0))
        screen.blit(label, (50, inp["rect"].y))
        pygame.draw.rect(screen, (0, 0, 0), inp["rect"], 2)
        pygame.draw.rect(screen, inp["color"], inp["rect"], 0)
        text_surface = font.render(inp["text"], True, (0, 0, 0))
        screen.blit(text_surface, (inp["rect"].x + 5, inp["rect"].y + 5))
    pygame.draw.rect(screen, button_color, button)
    screen.blit(button_text, (button.x + 15, button.y + 15))
    pygame.draw.rect(screen, quit_button_color, quit_button)
    screen.blit(quit_button_text, (quit_button.x + 15, quit_button.y + 15))

    # Display the result
    if h > 0:
        result_text = font.render("The convective heat transfer coefficient is %.2f W/m^2K." % h, True, (0, 0, 0))
        screen.blit(result_text, (50, int(height / 1.5)))

    # Update the display
    pygame.display.update()

# Quit Pygame
pygame.quit()
