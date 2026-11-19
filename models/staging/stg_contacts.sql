-- materialized: view
-- CRM contacts. Emails are normalised because the sales team types them by hand.
select
    contact_id,
    account_id as crm_account_id,
    lower(trim(email)) as email,
    split_part(lower(trim(email)), '@', 2) as email_domain,
    title,
    cast(created_at as timestamp) as created_at
from {{ source('raw', 'contacts') }}
