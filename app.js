const SCALER_MEAN = [86683.673, 562.863, 244478.740, 0.313, 0.363];
const SCALER_SCALE = [41579.090, 155.969, 140420.272, 0.464, 0.481];
const COEF = [0.8288, 2.4583, -0.9428, -0.0166, -2.1821];
const INTERCEPT = -0.9702;

// --- Currency conversion ---
// The model was trained on USD figures, so any amount entered in another
// currency has to be converted to USD before it's scored. Results are then
// converted back for display.
const CURRENCY_SYMBOLS = {
  USD: '$', EUR: '€', GBP: '£', INR: '₹', JPY: '¥', CAD: 'C$', AUD: 'A$'
};

// Fallback rates (units of currency per 1 USD), used only if the live
// rate fetch fails (e.g. offline).
const FALLBACK_RATES = {
  USD: 1, EUR: 0.92, GBP: 0.79, INR: 83.5, JPY: 149, CAD: 1.36, AUD: 1.52
};

let exchangeRates = { ...FALLBACK_RATES };

async function loadExchangeRates() {
  try {
    const targets = Object.keys(CURRENCY_SYMBOLS).filter(c => c !== 'USD').join(',');
    const res = await fetch(`https://api.frankfurter.app/latest?from=USD&to=${targets}`);
    if (!res.ok) throw new Error('Rate fetch failed');
    const data = await res.json();
    exchangeRates = { USD: 1, ...data.rates };
  } catch (err) {
    console.warn('Live exchange rates unavailable, using fallback rates.', err);
    exchangeRates = { ...FALLBACK_RATES };
  }
  updateCurrencySymbols();
}

function currentCurrency() {
  const el = document.getElementById('input-currency');
  return el ? el.value : 'USD';
}

function toUSD(amount, currencyCode) {
  const rate = exchangeRates[currencyCode] || 1;
  return amount / rate;
}

function formatCurrency(amount, currencyCode) {
  const symbol = CURRENCY_SYMBOLS[currencyCode] || '$';
  return symbol + Math.round(amount).toLocaleString();
}

function updateCurrencySymbols() {
  const code = currentCurrency();
  const symbol = CURRENCY_SYMBOLS[code] || '$';
  const incomeSymbol = document.getElementById('income-symbol');
  const loanSymbol = document.getElementById('loan-symbol');
  if (incomeSymbol) incomeSymbol.innerText = symbol;
  if (loanSymbol) loanSymbol.innerText = symbol;
}

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

document.getElementById('input-currency')?.addEventListener('change', updateCurrencySymbols);
loadExchangeRates();

document.getElementById('prediction-form').addEventListener('submit', function(e) {
  e.preventDefault();

  const currency = currentCurrency();
  const incomeEntered = parseFloat(document.getElementById('input-income').value);
  const credit = parseFloat(document.getElementById('input-credit').value);
  const loanEntered = parseFloat(document.getElementById('input-loan').value);
  const employment = document.getElementById('input-employment').value;

  // Convert to USD, since that's what the model was trained on.
  const income = toUSD(incomeEntered, currency);
  const loan = toUSD(loanEntered, currency);

  const result = predictApproval(income, credit, loan, employment);
  const isApproved = result.probApproved > 0.5;
  const confidencePct = (isApproved ? result.probApproved : 1 - result.probApproved) * 100;

  document.getElementById('result-empty').style.opacity = '0';
  setTimeout(() => {
    document.getElementById('result-empty').style.display = 'none';
    document.getElementById('result-content').style.opacity = '1';

    document.getElementById('sim-id').innerText = 'APL-' + Math.floor(Math.random() * 90000 + 10000);

    const usdNote = document.getElementById('usd-equivalent-note');
    if (usdNote) {
      if (currency === 'USD') {
        usdNote.innerText = '';
      } else {
        usdNote.innerText = `Scored as ${formatCurrency(income, 'USD')} income / ${formatCurrency(loan, 'USD')} loan (model runs in USD)`;
      }
    }

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
