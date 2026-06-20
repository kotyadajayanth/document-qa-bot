from query import ask_question

while True:

    question = input(
        "\nAsk Question (type exit to quit): "
    )

    if question.lower() == "exit":
        break

    ask_question(question)