from src.system_fit_calculator import SystemFitCalculator

def test_crash():
    print("Initializing Calculator...")
    calc = SystemFitCalculator()
    
    print("Testing with 'Arsenal'...")
    try:
        res = calc.calculate_fit({'dna_stamina':0.7}, "Arsenal")
        print(f"Result for Arsenal: {res}")
    except Exception as e:
        print(f"CRASHED on Arsenal: {e}")
        import traceback
        traceback.print_exc()

    print("Testing with 'Mikel Arteta'...")
    try:
        res = calc.calculate_fit({'dna_stamina':0.7}, "Mikel Arteta")
        print(f"Result for Arteta: {res}")
    except Exception as e:
        print(f"CRASHED on Arteta: {e}")
        
    print("Testing with 'Unknown'...")
    try:
        res = calc.calculate_fit({'dna_stamina':0.7}, "Unknown")
        print(f"Result for Unknown: {res}")
    except Exception as e:
        print(f"CRASHED on Unknown: {e}")

if __name__ == "__main__":
    test_crash()
