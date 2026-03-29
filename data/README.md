CSV format for LinkedIn/contact dataset

Place your CSV in `backend/data` and set `LINKEDIN_CSV_PATH` if using a custom path.

Supported column names (the loader will try these in order):
- `company`, `company_name`, `row_company`
- `experiences0company`, `experiences1company`
- `occupation`, `headline`, `summary`, `full_name`
- `name`, `full_name`, `first_name`, `last_name`
- `email`, `work_email`, `linkedin`, `linkedin_url`, `score`

Notes:
- The loader does a case-insensitive substring match on company names.
- If multiple rows match the company, the first several rows are used as leads.
- If no CSV matches are found, the tool falls back to synthesized/mock leads.
