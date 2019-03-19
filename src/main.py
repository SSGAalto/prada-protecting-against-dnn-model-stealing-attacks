import model
import modes


def main():
	oracle_path = "path/to/your/model"
	modes.serve_model(0.9, oracle_path, model.YourModel)


if __name__ == "__main__":
	main()
