-- Contact -> account resolution: trust the CRM link, otherwise fall back to email domain.
-- Personal-email contacts without a CRM link stay unresolved (an identity-resolution gap).
select
    c.contact_id,
    c.email,
    c.created_at,
    coalesce(c.crm_account_id, a.account_id) as account_id,
    case
        when c.crm_account_id is not null then 'crm'
        when a.account_id is not null then 'email_domain'
        else 'unresolved'
    end as resolution_method
from {{ ref('stg_contacts') }} as c
left join {{ ref('stg_accounts') }} as a on a.domain = c.email_domain
