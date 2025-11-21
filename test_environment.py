# import mlflow
# import numpy as np
# import pandas as pd
# import sklearn

# print("Environment is set up correctly!")
import requests
import json

# URL de l'API
API_URL = "http://localhost:8000"

def test_case(name, data, expected_result):
    """Tester un cas spécifique"""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"{'='*60}")
    
    try:
        response = requests.post(f"{API_URL}/predict", json=data)
        result = response.json()
        
        print(f"Données envoyées:")
        print(f"  - Revenu applicant: {data['ApplicantIncome']}")
        print(f"  - Revenu co-applicant: {data['CoapplicantIncome']}")
        print(f"  - Revenu total: {data['ApplicantIncome'] + data['CoapplicantIncome']}")
        print(f"  - Montant prêt: {data['LoanAmount']}")
        print(f"  - Historique crédit: {data['Credit_History']}")
        
        print(f"\nRésultat: {result['prediction']}")
        
        if result['prediction'] == expected_result:
            print(f"✅ TEST RÉUSSI - Résultat attendu: {expected_result}")
        else:
            print(f"❌ TEST ÉCHOUÉ - Attendu: {expected_result}, Obtenu: {result['prediction']}")
        
        # Afficher les détails
        if 'errors' in result and result['errors']:
            print(f"\nErreurs de validation:")
            for error in result['errors']:
                print(f"  - {error}")
        
        if 'financial_analysis' in result and result['financial_analysis']:
            fa = result['financial_analysis']
            print(f"\nAnalyse financière:")
            print(f"  - Revenu total: {fa['total_income']}")
            print(f"  - Montant prêt: {fa['loan_amount']}")
            print(f"  - Ratio prêt/revenu: {fa['loan_to_income_ratio']}")
            print(f"  - Statut crédit: {fa['credit_history_status']}")
        
        if 'probabilities' in result and result['probabilities']:
            print(f"\nProbabilités:")
            print(f"  - Rejet: {result['probabilities'][0]:.2%}")
            print(f"  - Approbation: {result['probabilities'][1]:.2%}")
        
        if 'nearest_neighbors' in result and result['nearest_neighbors']:
            nn = result['nearest_neighbors']
            if 'interpretation' in nn:
                print(f"\nVoisins proches: {nn['interpretation']}")
        
    except Exception as e:
        print(f"❌ ERREUR: {str(e)}")

def main():
    print("="*60)
    print("TESTS DE VALIDATION DU SYSTÈME DE PRÉDICTION DE PRÊTS")
    print("="*60)
    
    # Test 1: Cas aberrant original (devrait être rejeté)
    test_case(
        "Cas Aberrant - Revenu très faible, prêt énorme",
        {
            "Loan_ID": "TEST001",
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": "0",
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 50,
            "CoapplicantIncome": 0,
            "LoanAmount": 2000000,
            "Credit_History": 1.0,
            "Property_Area": "Urban"
        },
        expected_result="Rejected"
    )
    
    # Test 2: Cas aberrant avec prêt > 10000 (devrait être rejeté)
    test_case(
        "Prêt trop élevé (> 10000)",
        {
            "Loan_ID": "TEST002",
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": "0",
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 50000,
            "CoapplicantIncome": 10000,
            "LoanAmount": 15000,
            "Credit_History": 1.0,
            "Property_Area": "Urban"
        },
        expected_result="Rejected"
    )
    
    # Test 3: Revenu insuffisant pour le prêt (devrait être rejeté)
    test_case(
        "Revenu trop faible",
        {
            "Loan_ID": "TEST003",
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": "0",
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 500,
            "CoapplicantIncome": 0,
            "LoanAmount": 200,
            "Credit_History": 1.0,
            "Property_Area": "Urban"
        },
        expected_result="Rejected"
    )
    
    # Test 4: Cas réaliste bon (devrait être approuvé)
    test_case(
        "Bon profil - Approbation attendue",
        {
            "Loan_ID": "TEST004",
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": "0",
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 5000,
            "CoapplicantIncome": 2000,
            "LoanAmount": 150,
            "Credit_History": 1.0,
            "Property_Area": "Urban"
        },
        expected_result="Approved"
    )
    
    # Test 5: Mauvais historique crédit (peut être rejeté)
    test_case(
        "Mauvais historique de crédit",
        {
            "Loan_ID": "TEST005",
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": "0",
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 5000,
            "CoapplicantIncome": 2000,
            "LoanAmount": 150,
            "Credit_History": 0.0,
            "Property_Area": "Urban"
        },
        expected_result="Rejected"
    )
    
    # Test 6: Ratio prêt/revenu trop élevé
    test_case(
        "Ratio prêt/revenu trop élevé",
        {
            "Loan_ID": "TEST006",
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": "2",
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 2000,
            "CoapplicantIncome": 0,
            "LoanAmount": 500,
            "Credit_History": 1.0,
            "Property_Area": "Urban"
        },
        expected_result="Rejected"
    )
    
    # Test 7: Cas limite acceptable
    test_case(
        "Cas limite - Acceptable",
        {
            "Loan_ID": "TEST007",
            "Gender": "Female",
            "Married": "No",
            "Dependents": "0",
            "Education": "Not Graduate",
            "Self_Employed": "Yes",
            "ApplicantIncome": 3000,
            "CoapplicantIncome": 0,
            "LoanAmount": 100,
            "Credit_History": 1.0,
            "Property_Area": "Semiurban"
        },
        expected_result="Approved"
    )
    
    print(f"\n{'='*60}")
    print("FIN DES TESTS")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()