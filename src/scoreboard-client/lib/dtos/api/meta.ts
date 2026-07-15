export interface TableViewOption {
  key: string;
  label: string;
}

export interface DomainGroup {
  key: string;
  label: string;
  title: string;
}

export interface MetaResponse {
  auto_label: string;
  default_view: string;
  table_views: TableViewOption[];
  domain_groups: DomainGroup[];
  models: string[];
  model_choices: string[];
  entry_count: number;
  errors: string[];
}
