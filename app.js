const SCALER_MEAN = [86683.673, 562.863, 244478.740, 0.313, 0.363];
const SCALER_SCALE = [41579.090, 155.969, 140420.272, 0.464, 0.481];
const COEF = [0.8288, 2.4583, -0.9428, -0.0166, -2.1821];
const INTERCEPT = -0.9702;

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function predictApproval(income, credit, loan, employment) {
  const selfEmployed = employment === 'self' ? 1 : 0;
  const unemployed = employment === 'unemployed' ? 1 : 0;
  const raw = [income, credit, loan, selfEmployed, unemployed];

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
    creditContribution: contributions[1],
    incomeContribution: contributions[0],
    loanContribution: contributions[2],
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

    document.getElementById('driver-credit-label').innerText = impactLabel(creditAbs);
    document.getElementById('driver-credit-bar').style.width = barWidth(creditAbs);
    document.getElementById('driver-income-label').innerText = impactLabel(incomeAbs);
    document.getElementById('driver-income-bar').style.width = barWidth(incomeAbs);
    document.getElementById('driver-loan-label').innerText = impactLabel(loanAbs);
    document.getElementById('driver-loan-bar').style.width = barWidth(loanAbs);
  }, 300);
});
