import pygame
import csv

def run_simulation():
    import pygame
    screen_width = 1600
    screen_height = 800

    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    paused = False
    save_as = False

    # Define colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (128, 128, 128, 128)

    # Define fonts
    font = pygame.font.SysFont(None, 48)

    # Define buttons: Tip acess obj buttons[1]['rect']
    pause_buttons = [
        {"label": "Save", "rect": pygame.Rect(100, 100, 300, 50)},
        {"label": "Save As", "rect": pygame.Rect(100, 200, 300, 50)},
        {"label": "Iterator", "rect": pygame.Rect(100, 300, 300, 50)},
        {"label": "Quit to Main", "rect": pygame.Rect(100, 400, 300, 50)},
        {"label": "Quit to Desktop", "rect": pygame.Rect(100, 500, 300, 50)}
    ]
    game_state = {
    "score": 100,
    "level": 2,
    "player_position": (10, 20),}
    # Add other game state data here...
    # Game loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    paused = not paused
                    save_as = False

        screen.fill(WHITE)

        # Draw game elements here
        # Draw pause menu
        if paused:
            # This is a transparent grey background in the pause menu
            overlay = pygame.Surface((1600, 800))
            overlay.fill(GRAY)
            overlay.set_alpha(128)
            screen.blit(overlay, (0, 0))
            
            menu_text = font.render("Paused", True, BLACK)
            screen.blit(menu_text, (screen_width//2, 50))

            for i in range(len(pause_buttons)):
                pygame.draw.rect(screen, WHITE, pause_buttons[i]["rect"])
                button_text = font.render(pause_buttons[i]["label"], True, BLACK)
                if pause_buttons[i]["label"] == "Quit to Main":
                    screen.blit(button_text, pause_buttons[i]["rect"].move(100, 10))
                screen.blit(button_text, pause_buttons[i]["rect"].move(100, 10))
            #Handle Events
            if event.type == pygame.MOUSEBUTTONDOWN:
                if pause_buttons[0]['rect'].collidepoint(event.pos):
                    # Save Has been clicked
                    print("Game Saved")
                elif pause_buttons[1]['rect'].collidepoint(event.pos):
                    # Save as has been clicked
                    print("Save As")
                    save_as = True
                elif pause_buttons[2]['rect'].collidepoint(event.pos):
                    # Iterator has been pressed
                    print("Iterator has been pressed")
                elif pause_buttons[3]['rect'].collidepoint(event.pos):
                    # Quit to main has been pressed
                    print("Quit to main")
                    from main_menu import main_menu
                    main_menu()
                elif pause_buttons[4]['rect'].collidepoint(event.pos):
                    # Quit to desktop has been pressed
                    print("Quit to desktop")
                    pygame.quit()
            if save_as:
                # Create an input box for the filename
                input_rect = pygame.Rect(100, 300, 400, 50)
                input_text = ""
                # Draw the input box and text
                screen.blit(overlay, (0, 0))
                pygame.draw.rect(screen, WHITE, input_rect)
                pygame.display.update(input_rect)
                while save_as:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                save_as = False
                            elif event.key == pygame.K_RETURN:
                                # Save the file and return to the game
                                filename = input_text.strip() + ".csv"
                                with open(filename, "w", newline="") as f:
                                    writer = csv.writer(f)
                                    for key, value in game_state.items():
                                        writer.writerow([key, value])
                                print("Game saved as", filename)
                                save_as = False
                            elif event.key == pygame.K_BACKSPACE:
                                input_text = input_text[:-1]
                                input_surface = font.render(input_text, True, BLACK)
                                pygame.draw.rect(screen, WHITE, input_rect)
                                screen.blit(input_surface, (input_rect.x + 5, input_rect.y + 5))
                                pygame.draw.rect(screen, WHITE, input_rect, 2)
                                pygame.display.update(input_rect)

                            elif event.key not in (pygame.K_LSHIFT, pygame.K_RSHIFT):
                                input_text += event.unicode

                            input_surface = font.render(input_text, True, BLACK)
                            screen.blit(input_surface, (input_rect.x + 5, input_rect.y + 5))
                            pygame.draw.rect(screen, WHITE, input_rect, 2)
                            pygame.display.update(input_rect)

                    clock.tick(60)
                    continue

        pygame.display.flip()
        clock.tick(60)

    return
run_simulation()