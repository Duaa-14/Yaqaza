(function () {
  let manualRows = Array.isArray(window.__manualRows) ? window.__manualRows.slice() : [];

  // ---- source sub-tabs (upload / manual) ----
  document.querySelectorAll('.segmented-btn').forEach(function (btn) {
    btn.addEventListener('click', function () {
      document.querySelectorAll('.segmented-btn').forEach(function (b) { b.classList.remove('active'); });
      btn.classList.add('active');
      var target = btn.dataset.subtab;
      document.getElementById('panel-upload').classList.toggle('hidden', target !== 'upload');
      document.getElementById('panel-manual').classList.toggle('hidden', target !== 'manual');
    });
  });

  // ---- results tabs (invoice / profile) ----
  document.querySelectorAll('.result-tab-btn').forEach(function (btn) {
    btn.addEventListener('click', function () {
      document.querySelectorAll('.result-tab-btn').forEach(function (b) { b.classList.remove('active'); });
      btn.classList.add('active');
      var target = btn.dataset.restab;
      var invoicePanel = document.getElementById('restab-invoice');
      var profilePanel = document.getElementById('restab-profile');
      if (invoicePanel) invoicePanel.classList.toggle('hidden', target !== 'invoice');
      if (profilePanel) profilePanel.classList.toggle('hidden', target !== 'profile');
    });
  });

  // ---- file drop label text + drag feedback ----
  var fileInput = document.getElementById('fileInput');
  var fileDropText = document.getElementById('fileDropText');
  var fileDrop = document.getElementById('fileDrop');
  if (fileInput) {
    fileInput.addEventListener('change', function () {
      if (fileInput.files && fileInput.files.length > 0) {
        fileDropText.textContent = fileInput.files[0].name;
      } else {
        fileDropText.textContent = 'اسحب الملف هنا أو اضغط للاختيار';
      }
    });
  }
  if (fileDrop) {
    ['dragenter', 'dragover'].forEach(function (evt) {
      fileDrop.addEventListener(evt, function (e) {
        e.preventDefault();
        fileDrop.classList.add('dragover');
      });
    });
    ['dragleave', 'drop'].forEach(function (evt) {
      fileDrop.addEventListener(evt, function (e) {
        e.preventDefault();
        fileDrop.classList.remove('dragover');
      });
    });
  }

  // ---- manual entry table ----
  var tbody = document.getElementById('manualTbody');
  var emptyNote = document.getElementById('manualEmptyNote');
  var hiddenField = document.getElementById('manualRowsJson');
  var paymentLabels = { cash: 'نقدي', card: 'بطاقة' };

  function syncHiddenField() {
    hiddenField.value = JSON.stringify(manualRows);
  }

  function renderManualTable() {
    tbody.innerHTML = '';
    emptyNote.classList.toggle('hidden', manualRows.length > 0);
    manualRows.forEach(function (row, idx) {
      var tr = document.createElement('tr');
      tr.innerHTML =
        '<td>' + (idx + 1) + '</td>' +
        '<td>' + row.date + '</td>' +
        '<td>' + row.hour + '</td>' +
        '<td>' + row.amount + '</td>' +
        '<td>' + row.employee + '</td>' +
        '<td>' + (paymentLabels[row.payment] || row.payment) + '</td>' +
        '<td>' + row.discount_pct + '</td>' +
        '<td><button type="button" class="remove-row-btn" data-idx="' + idx + '">✕</button></td>';
      tbody.appendChild(tr);
    });
    syncHiddenField();
  }

  tbody.addEventListener('click', function (e) {
    var btn = e.target.closest('.remove-row-btn');
    if (!btn) return;
    var idx = parseInt(btn.dataset.idx, 10);
    manualRows.splice(idx, 1);
    renderManualTable();
  });

  document.getElementById('addManualBtn').addEventListener('click', function () {
    var invoiceId = document.getElementById('m_invoice_id').value || (manualRows.length + 1);
    var date = document.getElementById('m_date').value;
    var hour = document.getElementById('m_hour').value;
    var amount = document.getElementById('m_amount').value;
    var employee = document.getElementById('m_employee').value.trim();
    var payment = document.getElementById('m_payment').value;
    var discount = document.getElementById('m_discount').value;

    if (!date || !employee) {
      alert('الرجاء إدخال التاريخ واسم الموظف على الأقل.');
      return;
    }

    manualRows.push({
      invoice_id: parseInt(invoiceId, 10),
      date: date,
      hour: parseInt(hour, 10) || 0,
      amount: parseFloat(amount) || 0,
      employee: employee,
      payment: payment,
      discount_pct: parseInt(discount, 10) || 0,
    });
    renderManualTable();

    document.getElementById('m_invoice_id').value = parseInt(invoiceId, 10) + 1;
    document.getElementById('m_employee').value = '';
    document.getElementById('m_amount').value = 0;
    document.getElementById('m_discount').value = 0;
  });

  document.getElementById('clearManualBtn').addEventListener('click', function () {
    if (manualRows.length === 0) return;
    if (!confirm('هل تريد مسح جميع الفواتير المدخلة يدويًا؟')) return;
    manualRows = [];
    renderManualTable();
  });

  document.getElementById('downloadManualBtn').addEventListener('click', function () {
    if (manualRows.length === 0) {
      alert('لا توجد فواتير لتنزيلها.');
      return;
    }
    var cols = ['invoice_id', 'date', 'hour', 'amount', 'employee', 'payment', 'discount_pct'];
    var lines = [cols.join(',')];
    manualRows.forEach(function (row) {
      lines.push(cols.map(function (c) { return row[c]; }).join(','));
    });
    var blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = 'manual_invoices.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });

  renderManualTable();

  // ---- sensitivity slider ----
  var rateSlider = document.getElementById('rateSlider');
  var rateLabel = document.getElementById('rateLabel');
  var rateTimer = null;
  rateSlider.addEventListener('input', function () {
    rateLabel.textContent = rateSlider.value + '%';
  });
  rateSlider.addEventListener('change', function () {
    if (!window.__hasData) return;
    if (rateTimer) clearTimeout(rateTimer);
    rateTimer = setTimeout(function () {
      window.location.href = '/reanalyze/' + rateSlider.value;
    }, 150);
  });

  // ---- group-by select (profile level) ----
  var groupColSelect = document.getElementById('groupColSelect');
  groupColSelect.addEventListener('change', function () {
    if (!window.__hasData) return;
    window.location.href = '/profile/' + groupColSelect.value;
  });

  // ---- loading state (works for both the analyze button and the upload/map button) ----
  document.getElementById('analyzeForm').addEventListener('submit', function (e) {
    var submitter = e.submitter;
    if (submitter && submitter.id === 'uploadPreviewBtn') {
      submitter.disabled = true;
      submitter.textContent = 'جارٍ الرفع...';
    } else {
      var btn = document.getElementById('analyzeBtn');
      btn.disabled = true;
      btn.textContent = 'جارٍ التحليل...';
    }
  });
})();
