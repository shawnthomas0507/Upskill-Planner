from graph import app

while True:
    user_input = input("Enter your name (type 'quit' to exit): ")

    if user_input.lower() == 'quit':
        print("Goodbye!")
        break
    else:
        app.invoke({"name":user_input})
    