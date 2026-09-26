// Synced from loan_prediction.py's print_js_sync_constants().
// Feature order: [Applicant_Income, Credit_Score, Loan_Amount, Debt_to_Income, Is_Self-Employed, Is_Unemployed]
const SCALER_MEAN = [85818.429167, 559.177083, 251479.395833, 4.254690, 0.308333, 0.370833];
const SCALER_SCALE = [42240.976044, 155.899837, 140809.141919, 4.277155, 0.461805, 0.483028];
const COEF = [1.608054, 3.166321, -1.553622, 0.312045, -0.293842, -2.945177];
const INTERCEPT = -1.187694;

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function predictApproval(income, credit, loan, employment) {
  const selfEmployed = employment === 'self' ? 1 : 0;
  const unemployed = employment === 'unemployed' ? 1 : 0;
  const debtToIncome = loan / income;
  const raw = [income, credit, loan, debtToIncome, selfEmployed, unemployed];

  const scaled = raw.map((v, i) => (v - SCALER_MEAN[i]) / SCALER_SCALE[i]);

  let z = INTERCEPT;
  const contributions = [];
  for (let i = 0; i < scaled.length; i++) {
    const contribution = scaled[i] * COEF[i];
    z += contribution;
    contributions.push(contribution);
  }

  const probApproved = sigmoid(z);
  return {
    probApproved,
    incomeContribution: contributions[0],
    creditContribution: contributions[1],
    loanContribution: contributions[2],
    dtiContribution: contributions[3],
  };
}

function impactLabel(absContribution) {
  if (absContribution > 1.0) return 'High Impact';
  if (absContribution > 0.4) return 'Mod Impact';
  return 'Low Impact';
}

function barWidth(absContribution) {
  return Math.min(95, Math.max(10, absContribution * 40)) + '%';
}

document.getElementById('prediction-form').addEventListener('submit', function(e) {
  e.preventDefault();

  const income = parseFloat(document.getElementById('input-income').value);
  const credit = parseFloat(document.getElementById('input-credit').value);
  const loan = parseFloat(document.getElementById('input-loan').value);
  const employment = document.getElementById('input-employment').value;

  const result = predictApproval(income, credit, loan, employment);
  const isApproved = result.probApproved > 0.5;
  const confidencePct = (isApproved ? result.probApproved : 1 - result.probApproved) * 100;

  document.getElementById('result-empty').style.opacity = '0';
  setTimeout(() => {
    document.getElementById('result-empty').style.display = 'none';
    document.getElementById('result-content').style.opacity = '1';

    document.getElementById('sim-id').innerText = 'APL-' + Math.floor(Math.random() * 90000 + 10000);

    const badge = document.getElementById('decision-badge');
    const riskLevel = document.getElementById('risk-level');
    const riskBar = document.getElementById('risk-bar');
    const reason = document.getElementById('decision-reason');
    const conf = document.getElementById('conf-score');

    conf.innerText = confidencePct.toFixed(1);

    if (isApproved) {
      badge.innerText = 'Approved';
      badge.className = 'px-6 py-2 rounded-full font-body-bold text-[20px] bg-success-green/10 text-secondary border border-success-green/20 mb-2 transition-all';

      riskLevel.innerText = 'Low Risk';
      riskLevel.className = 'font-body-bold text-secondary';
      riskBar.className = 'bg-secondary h-2 rounded-full transition-all duration-1000';
      riskBar.style.width = (100 - confidencePct).toFixed(0) + '%';
      reason.innerText = "The model's learned coefficients favor approval for this applicant profile.";
    } else {
      badge.innerText = 'Rejected';
      badge.className = 'px-6 py-2 rounded-full font-body-bold text-[20px] bg-error-container text-error border border-error/20 mb-2 transition-all';

      riskLevel.innerText = 'High Risk';
      riskLevel.className = 'font-body-bold text-error';
      riskBar.className = 'bg-error h-2 rounded-full transition-all duration-1000';
      riskBar.style.width = confidencePct.toFixed(0) + '%';
      reason.innerText = "The model's learned coefficients weigh against approval for this applicant profile.";
    }

    const creditAbs = Math.abs(result.creditContribution);
    const incomeAbs = Math.abs(result.incomeContribution);
    const loanAbs = Math.abs(result.loanContribution);
    const dtiAbs = Math.abs(result.dtiContribution);

    document.getElementById('driver-credit-label').innerText = impactLabel(creditAbs);
    document.getElementById('driver-credit-bar').style.width = barWidth(creditAbs);
    document.getElementById('driver-income-label').innerText = impactLabel(incomeAbs);
    document.getElementById('driver-income-bar').style.width = barWidth(incomeAbs);
    document.getElementById('driver-loan-label').innerText = impactLabel(loanAbs);
    document.getElementById('driver-loan-bar').style.width = barWidth(loanAbs);
    document.getElementById('driver-dti-label').innerText = impactLabel(dtiAbs);
    document.getElementById('driver-dti-bar').style.width = barWidth(dtiAbs);
  }, 300);
});
